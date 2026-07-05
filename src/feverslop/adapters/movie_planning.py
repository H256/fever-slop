from __future__ import annotations

import json
import re
from dataclasses import replace
from math import ceil

from feverslop.domain.movie import CinematicShot, MovieActor, MovieBible, MovieContinuityRule, MovieLocation, StoryArch


class LLMMoviePlanner:
    def __init__(self, llm):
        self.llm = llm

    def generate_story_arch(self, *, title: str, source_type: str, story_text: str, desired_length: float) -> StoryArch:
        raw = self.llm.complete_prompt(
            _story_arch_prompt(title=title, source_type=source_type, story_text=story_text, desired_length=desired_length),
            system_prompt="You are a film writer. Return ONLY valid JSON.",
        )
        data = _json_object(raw)
        beats = data.get("beats") or []
        return StoryArch(
            title=str(data.get("title") or title),
            premise=str(data.get("premise") or story_text).strip(),
            beats=tuple(_beat_text(beat) for beat in beats if _beat_text(beat)),
        )

    def generate_movie_bible(self, *, title: str, source_type: str, story_text: str, desired_length: float, story_arch: StoryArch, config: dict) -> MovieBible:
        raw = self.llm.complete_prompt(
            _movie_bible_prompt(title=title, source_type=source_type, story_text=story_text, desired_length=desired_length, story_arch=story_arch, config=config),
            system_prompt="You are a film development producer. Return ONLY valid JSON.",
        )
        data = _json_object(raw)
        return _movie_bible_from_data(data, title=title, story_arch=story_arch, config=config, desired_length=desired_length)

    def generate_movie_continuity_plan(self, *, title: str, source_type: str, story_text: str, desired_length: float, bible: MovieBible, shots: tuple[CinematicShot, ...], config: dict) -> dict:
        raw = self.llm.complete_prompt(
            _movie_continuity_plan_prompt(title=title, source_type=source_type, story_text=story_text, desired_length=desired_length, bible=bible, shots=shots, config=config),
            system_prompt="You are a film continuity supervisor. Return ONLY valid JSON.",
        )
        return _json_object(raw)

    def generate_movie_screenplay(self, *, title: str, source_type: str, story_text: str, desired_length: float, bible: MovieBible, story_arch: StoryArch, config: dict) -> dict:
        raw = self.llm.complete_prompt(
            _movie_screenplay_prompt(title=title, source_type=source_type, story_text=story_text, desired_length=desired_length, bible=bible, story_arch=story_arch, config=config),
            system_prompt="You are a film screenwriter. Return ONLY valid JSON.",
        )
        return _json_object(raw)

    def generate_movie_narrative_plan(self, *, title: str, source_type: str, desired_length: float, bible: MovieBible, screenplay, config: dict) -> dict:
        raw = self.llm.complete_prompt(
            _movie_narrative_plan_prompt(title=title, source_type=source_type, desired_length=desired_length, bible=bible, screenplay=screenplay, config=config),
            system_prompt="You are a film story editor. Return ONLY valid JSON.",
        )
        return _json_object(raw)

    def plan_shots_from_bible(
        self,
        *,
        bible: MovieBible,
        desired_length: float,
        width: int,
        height: int,
        min_duration: float = 4.0,
        max_duration: float = 20.0,
    ) -> tuple[CinematicShot, ...]:
        raw = self.llm.complete_prompt(
            _shot_plan_from_bible_prompt(
                bible=bible,
                desired_length=desired_length,
                width=width,
                height=height,
                min_duration=min_duration,
                max_duration=max_duration,
            ),
            system_prompt="You are a film director and shot planner. Return ONLY valid JSON.",
        )
        data = _json_object(raw)
        shots = data.get("shots") or []
        if not isinstance(shots, list) or not shots:
            return DeterministicMoviePlanner().plan_shots_from_bible(
                bible=bible,
                desired_length=desired_length,
                width=width,
                height=height,
                min_duration=min_duration,
                max_duration=max_duration,
            )
        return _shots_from_data(
            shots,
            desired_length=desired_length,
            min_duration=min_duration,
            max_duration=max_duration,
        )

    def plan_shots(
        self,
        *,
        story_arch: StoryArch,
        desired_length: float,
        width: int,
        height: int,
        min_duration: float = 4.0,
        max_duration: float = 20.0,
    ) -> tuple[CinematicShot, ...]:
        raw = self.llm.complete_prompt(
            _shot_plan_prompt(
                story_arch=story_arch,
                desired_length=desired_length,
                width=width,
                height=height,
                min_duration=min_duration,
                max_duration=max_duration,
            ),
            system_prompt="You are a film director and shot planner. Return ONLY valid JSON.",
        )
        data = _json_object(raw)
        shots = data.get("shots") or []
        if not isinstance(shots, list) or not shots:
            return DeterministicMoviePlanner().plan_shots(
                story_arch=story_arch,
                desired_length=desired_length,
                width=width,
                height=height,
                min_duration=min_duration,
                max_duration=max_duration,
            )
        duration = max(1.0, float(desired_length) / len(shots))
        planned = []
        for index, raw_shot in enumerate(shots, start=1):
            shot = raw_shot if isinstance(raw_shot, dict) else {"description": str(raw_shot)}
            planned.append(
                CinematicShot(
                    shot_id=str(shot.get("shot_id") or f"shot_{index:04}"),
                    description=str(shot.get("description") or shot.get("action") or f"Shot {index}").strip(),
                    duration_seconds=float(shot.get("duration_seconds") or duration),
                    camera=str(shot.get("camera") or "motivated cinematic camera movement").strip(),
                    action=str(shot.get("action") or shot.get("description") or "").strip(),
                    expression=str(shot.get("expression") or "emotionally grounded performance").strip(),
                    location=str(shot.get("location") or "story-consistent cinematic location").strip(),
                    dialogue=str(shot.get("dialogue") or "").strip(),
                    actor_ids=tuple(_string_list(shot.get("actor_ids") or shot.get("actors"))),
                    location_id=_safe_id(shot.get("location_id") or shot.get("location")),
                )
            )
        planned = _ensure_minimum_actors(planned, story_arch)
        return _normalize_movie_shots(
            planned,
            desired_length=float(desired_length),
            min_duration=float(min_duration),
            max_duration=float(max_duration),
        )


class DeterministicMoviePlanner:
    def generate_story_arch(self, *, title: str, source_type: str, story_text: str, desired_length: float) -> StoryArch:
        text = " ".join(story_text.strip().split())
        beats = _split_screenplay_beats(story_text) if source_type == "screenplay" else _split_beats(text)
        return StoryArch(title=title, premise=text, beats=tuple(beats))

    def generate_movie_bible(self, *, title: str, source_type: str, story_text: str, desired_length: float, story_arch: StoryArch, config: dict) -> MovieBible:
        return _movie_bible_from_data({}, title=title, story_arch=story_arch, config=config, desired_length=desired_length)

    def generate_movie_continuity_plan(self, **_kwargs) -> dict:
        return {}

    def generate_movie_screenplay(self, **_kwargs) -> dict:
        return {}

    def generate_movie_narrative_plan(self, **_kwargs) -> dict:
        return {}

    def plan_shots_from_bible(
        self,
        *,
        bible: MovieBible,
        desired_length: float,
        width: int,
        height: int,
        min_duration: float = 4.0,
        max_duration: float = 20.0,
    ) -> tuple[CinematicShot, ...]:
        shots = self.plan_shots(
            story_arch=bible.story_arch,
            desired_length=desired_length,
            width=width,
            height=height,
            min_duration=min_duration,
            max_duration=max_duration,
        )
        default_actor = bible.actors[0].id if bible.actors else "main_character"
        default_location = bible.locations[0].id if bible.locations else "primary_location"
        default_location_name = bible.locations[0].name if bible.locations else "Primary Location"
        return tuple(
            replace(
                shot,
                actor_ids=shot.actor_ids or (default_actor,),
                location_id=shot.location_id or default_location,
                location=default_location_name if not shot.location or shot.location == "story-consistent cinematic location" else shot.location,
            )
            for shot in shots
        )

    def plan_shots(
        self,
        *,
        story_arch: StoryArch,
        desired_length: float,
        width: int,
        height: int,
        min_duration: float = 4.0,
        max_duration: float = 20.0,
    ) -> tuple[CinematicShot, ...]:
        beats = story_arch.beats or (story_arch.premise,)
        duration = max(1.0, float(desired_length) / len(beats))
        shots = []
        for index, beat in enumerate(beats, start=1):
            screenplay = _parse_screenplay_beat(beat)
            if screenplay is not None:
                description = screenplay["action"] or screenplay["dialogue"] or screenplay["location"]
                shots.append(
                    CinematicShot(
                        shot_id=f"shot_{index:04}",
                        description=description,
                        duration_seconds=duration,
                        camera=screenplay["camera"],
                        action=screenplay["action"],
                        expression=screenplay["expression"],
                    location=screenplay["location"],
                    dialogue=screenplay["dialogue"],
                    actor_ids=tuple(_dialogue_actor_ids(screenplay["dialogue"])),
                    location_id=_safe_id(screenplay["location"]),
                )
                )
                continue
            shots.append(
                CinematicShot(
                    shot_id=f"shot_{index:04}",
                    description=beat,
                    duration_seconds=duration,
                    camera="slow dolly with motivated cinematic framing",
                    action=beat,
                    expression="subtle emotionally grounded acting",
                    location="story-consistent cinematic location",
                    actor_ids=(),
                    location_id="",
                )
            )
        shots = _ensure_minimum_actors(shots, story_arch)
        return _normalize_movie_shots(
            shots,
            desired_length=float(desired_length),
            min_duration=float(min_duration),
            max_duration=float(max_duration),
        )


def _movie_bible_from_data(data: dict, *, title: str, story_arch: StoryArch, config: dict, desired_length: float) -> MovieBible:
    actors = _configured_actors(config) or _actors_from_data(data.get("actors") or [])
    locations = _configured_locations(config) or _locations_from_data(data.get("locations") or [])
    if not actors:
        actors = [MovieActor(id="main_character", name="Main Character", role="lead", visual_description="story-defined cinematic lead character with consistent face, hair, wardrobe, posture, and body shape")]
    if not locations:
        locations = [MovieLocation(id="primary_location", name="Primary Location", visual_description="story-defined cinematic location with consistent production design, geography, lighting, and atmosphere")]
    return MovieBible(
        title=str(data.get("title") or title),
        premise=str(data.get("premise") or story_arch.premise).strip(),
        story_arch=story_arch,
        actors=tuple(actors),
        locations=tuple(locations),
        continuity=tuple(_continuity_from_data(data.get("continuity") or []))
        or (MovieContinuityRule(id="visual_continuity", description="Keep actor identities, wardrobe, locations, lighting logic, and story geography consistent across shots."),),
        style_constraints=tuple(str(item).strip() for item in data.get("style_constraints") or _config_style_constraints(config) if str(item).strip()),
        runtime_constraints={
            "desired_length": float(desired_length),
            "max_scene_actors": min(4, max(1, int(config.get("max_scene_actors") or 4))),
            **({"fps": int(config["fps"])} if config.get("fps") else {}),
            **({"dialogue_language": _config_dialogue_language(config)} if _config_dialogue_language(config) else {}),
        },
    )


def _configured_actors(config: dict) -> list[MovieActor]:
    actors = []
    raw = config.get("actors") if isinstance(config.get("actors"), list) else []
    for index, actor in enumerate(raw, start=1):
        if not isinstance(actor, dict):
            continue
        actors.append(
            MovieActor(
                id=_safe_id(actor.get("id") or actor.get("name")) or f"actor_{index}",
                name=str(actor.get("name") or actor.get("id") or f"Actor {index}").strip(),
                role=str(actor.get("role") or "").strip(),
                visual_description=str(actor.get("visual_description") or actor.get("image_prompt") or actor.get("prompt") or actor.get("name") or f"Actor {index}").strip(),
            )
        )
    return actors


def _configured_locations(config: dict) -> list[MovieLocation]:
    raw = config.get("structured_locations")
    if not isinstance(raw, list) or not raw:
        raw = config.get("locations") if isinstance(config.get("locations"), list) else []
    locations = []
    for index, location in enumerate(raw, start=1):
        if isinstance(location, dict):
            locations.append(
                MovieLocation(
                    id=_safe_id(location.get("id") or location.get("name")) or f"location_{index}",
                    name=str(location.get("name") or location.get("id") or f"Location {index}").strip(),
                    visual_description=str(location.get("visual_description") or location.get("image_prompt") or location.get("prompt") or location.get("name") or f"Location {index}").strip(),
                )
            )
        elif str(location or "").strip():
            name = str(location).strip()
            locations.append(MovieLocation(id=_safe_id(name) or f"location_{index}", name=name, visual_description=name))
    return locations


def _actors_from_data(raw: object) -> list[MovieActor]:
    actors = []
    for index, actor in enumerate(raw if isinstance(raw, list) else [], start=1):
        if not isinstance(actor, dict):
            continue
        actors.append(
            MovieActor(
                id=_safe_id(actor.get("id") or actor.get("name")) or f"actor_{index}",
                name=str(actor.get("name") or actor.get("id") or f"Actor {index}").strip(),
                role=str(actor.get("role") or "").strip(),
                visual_description=str(actor.get("visual_description") or actor.get("description") or actor.get("name") or f"Actor {index}").strip(),
            )
        )
    return actors


def _locations_from_data(raw: object) -> list[MovieLocation]:
    locations = []
    for index, location in enumerate(raw if isinstance(raw, list) else [], start=1):
        if not isinstance(location, dict):
            continue
        locations.append(
            MovieLocation(
                id=_safe_id(location.get("id") or location.get("name")) or f"location_{index}",
                name=str(location.get("name") or location.get("id") or f"Location {index}").strip(),
                visual_description=str(location.get("visual_description") or location.get("description") or location.get("name") or f"Location {index}").strip(),
            )
        )
    return locations


def _continuity_from_data(raw: object) -> list[MovieContinuityRule]:
    rules = []
    for index, rule in enumerate(raw if isinstance(raw, list) else [], start=1):
        if isinstance(rule, dict):
            description = str(rule.get("description") or rule.get("rule") or "").strip()
            rule_id = _safe_id(rule.get("id") or description) or f"continuity_{index}"
        else:
            description = str(rule).strip()
            rule_id = _safe_id(description) or f"continuity_{index}"
        if description:
            rules.append(MovieContinuityRule(id=rule_id, description=description))
    return rules


def _config_style_constraints(config: dict) -> list[str]:
    values = []
    for key in ("subject", "style", "prompt_guidance"):
        value = config.get(key)
        if value:
            values.append(json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value))
    return values


def _config_dialogue_language(config: dict) -> str:
    return str(config.get("dialogue_language") or "").strip()


def _shots_from_data(shots: list, *, desired_length: float, min_duration: float, max_duration: float) -> tuple[CinematicShot, ...]:
    duration = max(1.0, float(desired_length) / len(shots))
    planned = []
    for index, raw_shot in enumerate(shots, start=1):
        shot = raw_shot if isinstance(raw_shot, dict) else {"description": str(raw_shot)}
        planned.append(
            CinematicShot(
                shot_id=str(shot.get("shot_id") or f"shot_{index:04}"),
                description=str(shot.get("description") or shot.get("action") or f"Shot {index}").strip(),
                duration_seconds=float(shot.get("duration_seconds") or duration),
                camera=str(shot.get("camera") or "motivated cinematic camera movement").strip(),
                action=str(shot.get("action") or shot.get("description") or "").strip(),
                expression=str(shot.get("acting") or shot.get("expression") or "emotionally grounded performance").strip(),
                location=str(shot.get("location") or "story-consistent cinematic location").strip(),
                dialogue=str(shot.get("dialogue") or "").strip(),
                actor_ids=tuple(_string_list(shot.get("actor_ids") or shot.get("actors"))),
                location_id=_safe_id(shot.get("location_id") or shot.get("location")),
            )
        )
    return _normalize_movie_shots(
        planned,
        desired_length=float(desired_length),
        min_duration=float(min_duration),
        max_duration=float(max_duration),
    )


def _split_beats(text: str) -> list[str]:
    parts = [part.strip(" .!?") for part in re.split(r"[.!?]+", text.replace("\n", " ")) if part.strip()]
    return parts[:12] or [text]


_HEADING_RE = re.compile(r"^(INT\.|EXT\.|INT/EXT\.)\s+(.+)$", re.IGNORECASE)


def _split_screenplay_beats(text: str) -> list[str]:
    beats: list[str] = []
    heading = ""
    body: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _HEADING_RE.match(line)
        if match:
            if heading:
                beats.append("\n".join([heading, *body]))
            heading = f"{match.group(1).upper()} {match.group(2).strip()}"
            body = []
        elif heading:
            body.append(line)
    if heading:
        beats.append("\n".join([heading, *body]))
    return beats or _split_beats(" ".join(text.strip().split()))


def _parse_screenplay_beat(beat: str) -> dict[str, str] | None:
    lines = [line.strip() for line in beat.splitlines() if line.strip()]
    if not lines:
        return None
    heading = _HEADING_RE.match(lines[0])
    if heading is None:
        return None

    kind = heading.group(1).upper()
    location = heading.group(2).strip()
    dialogue: list[str] = []
    actions: list[str] = []
    index = 1
    while index < len(lines):
        line = lines[index]
        if _is_character_cue(line) and index + 1 < len(lines):
            dialogue.append(f"{line}: {lines[index + 1]}")
            index += 2
            continue
        actions.append(line)
        index += 1

    action = " ".join(actions).strip()
    camera = "controlled interior dolly with motivated cinematic framing" if kind.startswith("INT") else "wide exterior establishing move with cinematic depth"
    return {
        "location": location,
        "dialogue": " ".join(dialogue),
        "action": action,
        "camera": camera,
        "expression": "emotion and facial acting follow the dialogue beats" if dialogue else "subtle emotionally grounded acting",
    }


def _is_character_cue(line: str) -> bool:
    words = line.split()
    return bool(words) and len(words) <= 4 and line.upper() == line and not _HEADING_RE.match(line)


def _story_arch_prompt(*, title: str, source_type: str, story_text: str, desired_length: float) -> str:
    return f"""
Create a movie story arch from this {source_type}.
Title: {title}
Target duration seconds: {desired_length}

Return JSON with:
{{"title": string, "premise": string, "beats": [string]}}

Source:
{story_text}
""".strip()


def _shot_plan_prompt(*, story_arch: StoryArch, desired_length: float, width: int, height: int, min_duration: float, max_duration: float) -> str:
    target_shots = max(1, ceil(float(desired_length) / max(1.0, min(float(max_duration), 12.0))))
    return f"""
Create a continuous cinematic shot plan from this story arch.
Title: {story_arch.title}
Premise: {story_arch.premise}
Beats: {json.dumps(list(story_arch.beats), ensure_ascii=False)}
Target duration seconds: {desired_length}
Resolution: {width}x{height}
Target shot count: about {target_shots}. Prefer varied shot durations from {min_duration:g} to {max_duration:g} seconds. Never exceed {max_duration:g} seconds for one shot.

Return JSON with:
{{"shots": [{{"description": string, "duration_seconds": number, "camera": string, "action": string, "expression": string, "location": string, "dialogue": string, "actor_ids": [string], "location_id": string}}]}}

Rules:
- If the source or steering names actors/characters, preserve them as stable snake_case actor_ids.
- If the idea asks for at least N characters, create at least N distinct actor_ids across the shot plan.
- Use stable snake_case location_id values for recurring locations.
""".strip()


def _movie_bible_prompt(*, title: str, source_type: str, story_text: str, desired_length: float, story_arch: StoryArch, config: dict) -> str:
    dialogue_language = _config_dialogue_language(config)
    dialogue_rule = f"- All dialogue in the movie bible and downstream shot plan must be in {dialogue_language} only." if dialogue_language else ""
    return f"""
Create a movie bible for this {source_type}.
Title: {title}
Target duration seconds: {desired_length}
Dialogue language: {dialogue_language or "unspecified"}
Story arch: {json.dumps({"title": story_arch.title, "premise": story_arch.premise, "beats": list(story_arch.beats)}, ensure_ascii=False)}
Config constraints: {json.dumps(config, ensure_ascii=False)}

Return JSON with:
{{"title": string, "premise": string, "actors": [{{"id": snake_case, "name": string, "role": string, "visual_description": string}}], "locations": [{{"id": snake_case, "name": string, "visual_description": string}}], "continuity": [{{"id": snake_case, "description": string}}], "style_constraints": [string]}}

Rules:
- If config.actors is present, use exactly those actor ids and do not invent actor ids.
- If config.structured_locations or config.locations is present, use exactly those location ids and do not invent location ids.
- Actor and location visual_description must describe stable visual identity only, not camera moves, shots, dialogue, or reference-sheet layout.
- Preserve screenplay dialogue cues in continuity/story structure, not in visual descriptions.
{dialogue_rule}

Source:
{story_text}
""".strip()


def _shot_plan_from_bible_prompt(*, bible: MovieBible, desired_length: float, width: int, height: int, min_duration: float, max_duration: float) -> str:
    target_shots = max(1, ceil(float(desired_length) / max(1.0, min(float(max_duration), 12.0))))
    dialogue_language = str((bible.runtime_constraints or {}).get("dialogue_language") or "").strip()
    dialogue_rule = f"- Every dialogue field must be written in {dialogue_language} only. Do not mix in any other spoken language." if dialogue_language else ""
    return f"""
Create a continuous cinematic render plan from this movie bible.
Bible: {json.dumps({
        "title": bible.title,
        "premise": bible.premise,
        "beats": list(bible.story_arch.beats),
        "actors": [asdict_like_actor(actor) for actor in bible.actors],
        "locations": [asdict_like_location(location) for location in bible.locations],
        "continuity": [rule.description for rule in bible.continuity],
        "style_constraints": list(bible.style_constraints),
    }, ensure_ascii=False)}
Target duration seconds: {desired_length}
Dialogue language: {dialogue_language or "unspecified"}
Resolution: {width}x{height}
Target shot count: about {target_shots}. Prefer varied shot durations from {min_duration:g} to {max_duration:g} seconds. Never exceed {max_duration:g} seconds for one shot.

Return JSON with:
{{"shots": [{{"description": string, "duration_seconds": number, "camera": string, "action": string, "acting": string, "location": string, "dialogue": string, "actor_ids": [string], "location_id": string, "continuity_notes": string}}]}}

Rules:
- actor_ids must only use bible actor ids.
- location_id must only use bible location ids.
- Never put more than 4 actors in one shot.
- Preserve dialogue, camera, acting, action, and continuity as separate fields.
{dialogue_rule}
""".strip()


def _movie_continuity_plan_prompt(*, title: str, source_type: str, story_text: str, desired_length: float, bible: MovieBible, shots: tuple[CinematicShot, ...], config: dict) -> str:
    dialogue_language = str((bible.runtime_constraints or {}).get("dialogue_language") or "").strip()
    screenplay_rule = "- Source is a screenplay: preserve scene order, dialogue cues, and character cues. Do not rewrite the screenplay structure." if source_type == "screenplay" else "- Source is an idea/short story: create a stronger causal chain while preserving the premise and constraints."
    return f"""
Build a movie continuity plan for AI film generation.

Return valid JSON only. No markdown, no commentary.

Required top-level shape:
{{
  "continuity_ledger": {{
    "style_bible": {{"visual_style": "", "palette": "", "lighting": "", "camera": "", "negative_constraints": []}},
    "characters": {{"actor_id": {{"character_id": "", "base_identity": "", "wardrobe": "", "carried_props": [], "physical_state": "", "emotional_state": "", "last_location": "", "last_action": ""}}}},
    "locations": {{"location_id": {{"location_id": "", "name": "", "time_of_day": "", "lighting": "", "props": [], "environmental_state": ""}}}},
    "scene_order": []
  }},
  "scene_continuity": {{
    "shot_id": {{"shot_id": "", "location_id": "", "incoming": [], "required_carryovers": [], "allowed_changes": [], "outgoing": [], "characters": {{}}, "location": {{}}}}
  }},
  "narrative_chain": [
    {{"shot_id": "", "story_state_before": "", "story_state_after": "", "cause_from_previous": "", "narrative_purpose": "", "conflict_or_tension": "", "turning_point": "", "sets_up_next": ""}}
  ]
}}

Rules:
- Keep actor ids exactly to bible actor ids. Keep location ids exactly to bible location ids.
- Every shot id from SHOTS must appear once in scene_order, scene_continuity, and narrative_chain.
- Every shot after the first needs a concrete cause_from_previous.
- Every shot except the last needs a concrete sets_up_next.
- required_carryovers are facts the generator must preserve in this shot.
- allowed_changes are facts allowed or expected to change in this shot.
- outgoing states become useful incoming state for later shots.
- Never invent more than {int((bible.runtime_constraints or {}).get("max_scene_actors") or 4)} visible actors in one shot.
- Dialogue language is {dialogue_language or "unspecified"}; any dialogue continuity must respect it.
{screenplay_rule}

TITLE: {title}
TARGET DURATION: {desired_length}
CONFIG CONSTRAINTS: {json.dumps(config, ensure_ascii=False)}
BIBLE: {json.dumps({
        "title": bible.title,
        "premise": bible.premise,
        "actors": [asdict_like_actor(actor) for actor in bible.actors],
        "locations": [asdict_like_location(location) for location in bible.locations],
        "continuity": [rule.description for rule in bible.continuity],
        "style_constraints": list(bible.style_constraints),
    }, ensure_ascii=False)}
SHOTS: {json.dumps([{
        "shot_id": shot.shot_id,
        "description": shot.description,
        "action": shot.action,
        "camera": shot.camera,
        "acting": shot.expression,
        "dialogue": shot.dialogue,
        "actor_ids": list(shot.actor_ids),
        "location_id": shot.location_id,
        "location": shot.location,
    } for shot in shots], ensure_ascii=False)}
SOURCE:
{story_text}
""".strip()


def _movie_screenplay_prompt(*, title: str, source_type: str, story_text: str, desired_length: float, bible: MovieBible, story_arch: StoryArch, config: dict) -> str:
    dialogue_language = str((bible.runtime_constraints or {}).get("dialogue_language") or config.get("dialogue_language") or "").strip()
    screenplay_rule = "Preserve source scene order and dialogue exactly; annotate it into structured scenes." if source_type == "screenplay" else "Write a compact screenplay structure from the idea without exceeding the target duration."
    return f"""
Create the canonical structured screenplay for this movie.

Return JSON with:
{{"title": string, "source_type": "{source_type}", "dialogue_language": string, "scenes": [{{"scene_id": "scene_0001", "heading": string, "summary": string, "action": string, "dialogue": string, "actor_ids": [string], "location_id": string, "source_span": string}}]}}

Rules:
- {screenplay_rule}
- actor_ids must only use these ids: {[actor.id for actor in bible.actors]}
- location_id must only use these ids: {[location.id for location in bible.locations]}
- Dialogue language is {dialogue_language or "unspecified"}.
- Keep screenplay text and dialogue out of actor/location visual descriptions.

Title: {title}
Target duration seconds: {desired_length}
Story arch: {json.dumps({"title": story_arch.title, "premise": story_arch.premise, "beats": list(story_arch.beats)}, ensure_ascii=False)}
Bible: {json.dumps({"actors": [asdict_like_actor(actor) for actor in bible.actors], "locations": [asdict_like_location(location) for location in bible.locations]}, ensure_ascii=False)}
Config: {json.dumps(config, ensure_ascii=False)}
Source:
{story_text}
""".strip()


def _movie_narrative_plan_prompt(*, title: str, source_type: str, desired_length: float, bible: MovieBible, screenplay, config: dict) -> str:
    scenes = [
        {
            "scene_id": scene.scene_id,
            "summary": scene.summary,
            "action": scene.action,
            "dialogue": scene.dialogue,
            "actor_ids": list(scene.actor_ids),
            "location_id": scene.location_id,
        }
        for scene in getattr(screenplay, "scenes", ())
    ]
    return f"""
Create a narrative memory plan from the canonical screenplay.

Return JSON with:
{{"title": string, "sequences": [{{"sequence_id": string, "title": string, "scene_ids": [string], "dramatic_function": string}}], "causal_chain": [{{"scene_id": string, "story_state_before": string, "story_state_after": string, "cause_from_previous": string, "sets_up_next": string}}], "open_threads": [string]}}

Rules:
- Use only scene_id values from SCREENPLAY.
- Preserve scene order.
- Every scene after the first needs a concrete cause_from_previous.
- Every scene except the last needs a concrete sets_up_next.
- This is planning memory only; do not write renderer prompt prose.

Title: {title}
Source type: {source_type}
Target duration seconds: {desired_length}
Bible actors: {[actor.id for actor in bible.actors]}
Bible locations: {[location.id for location in bible.locations]}
Config: {json.dumps(config, ensure_ascii=False)}
SCREENPLAY: {json.dumps(scenes, ensure_ascii=False)}
""".strip()


def asdict_like_actor(actor: MovieActor) -> dict:
    return {"id": actor.id, "name": actor.name, "role": actor.role, "visual_description": actor.visual_description}


def asdict_like_location(location: MovieLocation) -> dict:
    return {"id": location.id, "name": location.name, "visual_description": location.visual_description}


def _json_object(raw: str) -> dict:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Movie planner LLM response must be a JSON object")
    return data


def _beat_text(beat) -> str:
    if isinstance(beat, dict):
        return str(beat.get("summary") or beat.get("description") or beat.get("beat") or "").strip()
    return str(beat).strip()


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [_safe_id(item) for item in value if _safe_id(item)]
    if isinstance(value, str) and value.strip():
        return [_safe_id(value)]
    return []


def _dialogue_actor_ids(value: str) -> list[str]:
    ids = []
    for part in str(value or "").split():
        if part.endswith(":"):
            actor_id = _safe_id(part[:-1])
            if actor_id and actor_id not in ids:
                ids.append(actor_id)
    return ids


def _safe_id(value: object) -> str:
    raw = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return raw


def _ensure_minimum_actors(shots: list[CinematicShot], story_arch: StoryArch) -> list[CinematicShot]:
    minimum = _minimum_actor_count(story_arch)
    if minimum <= 0 or not shots:
        return shots
    actor_ids: list[str] = []
    for shot in shots:
        for actor_id in shot.actor_ids:
            if actor_id and actor_id not in actor_ids:
                actor_ids.append(actor_id)
    while len(actor_ids) < minimum:
        actor_ids.append(f"character_{len(actor_ids) + 1}")
    updated = []
    for index, shot in enumerate(shots):
        needed = actor_ids[index::len(shots)] if len(shots) > 1 else actor_ids
        merged = tuple(dict.fromkeys([*shot.actor_ids, *needed]))
        updated.append(replace(shot, actor_ids=merged))
    return updated


def _minimum_actor_count(story_arch: StoryArch) -> int:
    text = " ".join([story_arch.premise, *story_arch.beats]).lower()
    word_numbers = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
    }
    pattern = r"at least\s+(\d+|one|two|three|four|five|six)\s+(?:distinct\s+)?(?:actors|characters|people|persons)"
    match = re.search(pattern, text)
    if not match:
        return 0
    value = match.group(1)
    return int(value) if value.isdigit() else word_numbers.get(value, 0)


def _normalize_movie_shots(shots: list[CinematicShot], *, desired_length: float, min_duration: float, max_duration: float) -> tuple[CinematicShot, ...]:
    if not shots:
        return tuple(shots)
    min_duration = max(1.0, float(min_duration))
    max_duration = max(min_duration, float(max_duration))
    expanded: list[CinematicShot] = []
    for shot in shots:
        duration = max(1.0, float(shot.duration_seconds))
        parts = max(1, ceil(duration / max_duration))
        for part in range(parts):
            expanded.append(
                replace(
                    shot,
                    shot_id=f"{shot.shot_id}_{part + 1}" if parts > 1 else shot.shot_id,
                    duration_seconds=duration / parts,
                )
            )
    pattern = (0.86, 1.08, 0.94, 1.18, 1.0, 0.78, 1.12)
    weighted = [max(min_duration, min(max_duration, shot.duration_seconds * pattern[index % len(pattern)])) for index, shot in enumerate(expanded)]
    total = sum(weighted) or 1.0
    scaled = [max(min_duration, min(max_duration, value * float(desired_length) / total)) for value in weighted]
    return tuple(replace(shot, shot_id=f"shot_{index:04}", duration_seconds=round(duration, 3)) for index, (shot, duration) in enumerate(zip(expanded, scaled), start=1))
