from __future__ import annotations

import json
import re
from dataclasses import replace
from math import ceil

from feverslop.domain.movie import CinematicShot, StoryArch


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
