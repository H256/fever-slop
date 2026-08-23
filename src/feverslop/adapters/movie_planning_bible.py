from __future__ import annotations

import json
import re
from dataclasses import replace

from feverslop.domain.movie import (
    MovieActor,
    MovieBible,
    MovieContinuityRule,
    MovieLocation,
    StoryArch,
)
from feverslop.domain.movie_utils import (
    clean_visual_description,
    configured_actors,
    configured_locations,
    display_name,
    safe_id,
)
from feverslop.domain.screenplay import (
    HEADING_RE,
    parse_screenplay,
    split_screenplay_dialogue,
)


def _movie_bible_from_data(data: dict, *, title: str, source_type: str = "", source_text: str = "", story_arch: StoryArch, config: dict, desired_length: float) -> MovieBible:
    configured_actor_list = configured_actors(config)
    configured_location_list = configured_locations(config)
    reference_story_arch = _screenplay_reference_arch(story_arch, source_type=source_type, source_text=source_text)
    story_actors = _actors_from_story_arch(reference_story_arch)
    story_locations = _locations_from_story_arch(reference_story_arch)
    data_actors = _actors_from_data(data.get("actors") or [])
    data_locations = _locations_from_data(data.get("locations") or [])
    actors = configured_actor_list or _merge_screenplay_references(story_actors, data_actors) or data_actors
    locations = configured_location_list or _merge_screenplay_references(story_locations, data_locations) or data_locations
    if not actors:
        actors = [MovieActor(id="main_character", name="Main Character", role="lead", visual_description="Main Character")]
    if not locations:
        locations = [MovieLocation(id="primary_location", name="Primary Location", visual_description="Primary Location")]
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


def _screenplay_reference_arch(story_arch: StoryArch, *, source_type: str, source_text: str) -> StoryArch:
    if source_type != "screenplay":
        return story_arch
    beats = _split_screenplay_beats(source_text)
    if not beats:
        return story_arch
    return replace(story_arch, beats=tuple(beats))


def _configured_actors(config: dict) -> list[MovieActor]:
    return configured_actors(config)


def _configured_locations(config: dict) -> list[MovieLocation]:
    return configured_locations(config)


def _actors_from_data(raw) -> list[MovieActor]:
    actors = []
    for index, actor in enumerate(raw if isinstance(raw, list) else [], start=1):
        if not isinstance(actor, dict):
            continue
        actors.append(
            MovieActor(
                id=safe_id(actor.get("id") or actor.get("name")) or f"actor_{index}",
                name=str(actor.get("name") or actor.get("id") or f"Actor {index}").strip(),
                role=str(actor.get("role") or "").strip(),
                visual_description=clean_visual_description(actor.get("visual_description") or actor.get("description"), str(actor.get("name") or actor.get("id") or f"Actor {index}").strip()),
            ),
        )
    return actors


def _locations_from_data(raw) -> list[MovieLocation]:
    locations = []
    for index, location in enumerate(raw if isinstance(raw, list) else [], start=1):
        if not isinstance(location, dict):
            continue
        locations.append(
            MovieLocation(
                id=safe_id(location.get("id") or location.get("name")) or f"location_{index}",
                name=str(location.get("name") or location.get("id") or f"Location {index}").strip(),
                visual_description=clean_visual_description(location.get("visual_description") or location.get("description"), str(location.get("name") or location.get("id") or f"Location {index}").strip()),
            ),
        )
    return locations


def _actors_from_story_arch(story_arch: StoryArch) -> list[MovieActor]:
    actors = []
    known_ids = set()
    for beat in story_arch.beats:
        parsed = _parse_screenplay_beat(beat)
        if parsed is None:
            continue
        for cue in _dialogue_actor_names(parsed["dialogue"]):
            actor_id = safe_id(cue)
            if not actor_id or actor_id in known_ids:
                continue
            name = display_name(cue)
            actors.append(MovieActor(id=actor_id, name=name, role="character", visual_description=name))
            known_ids.add(actor_id)
    return actors


def _locations_from_story_arch(story_arch: StoryArch) -> list[MovieLocation]:
    locations = []
    known_bases = set()
    character_names = set()
    for beat in story_arch.beats:
        parsed = _parse_screenplay_beat(beat)
        if parsed is None:
            continue
        for cue in _dialogue_actor_names(parsed["dialogue"]):
            character_names.add(cue)
        name = parsed["location"].strip()
        location_id = safe_id(name)
        base_id = safe_id(_location_base_name(name))
        if not location_id or not base_id:
            continue
        if _location_base_collides(base_id, known_bases):
            continue
        locations.append(MovieLocation(id=location_id, name=name, visual_description=_location_visual_description(name, parsed["action"], character_names=tuple(character_names))))
        known_bases.add(base_id)
    return locations


def _location_base_collides(base_id: str, known_bases: set[str]) -> bool:
    if base_id in known_bases:
        return True
    for known in known_bases:
        if base_id in known or known in base_id:
            return True
    return False


def _merge_screenplay_references(story_items, data_items) -> list:
    if not story_items:
        return []
    by_id = {item.id: item for item in story_items}
    matched_data_ids = set()
    merged = []
    for story_item in story_items:
        data_item = next((item for item in data_items if item.id == story_item.id), None)
        if data_item is None:
            story_name = getattr(story_item, "name", "")
            for candidate in data_items:
                if candidate.id in matched_data_ids:
                    continue
                candidate_name = getattr(candidate, "name", "")
                if _location_id_matches(story_item.id, candidate.id, name_a=story_name, name_b=candidate_name):
                    data_item = candidate
                    break
        if data_item is None:
            merged.append(story_item)
            continue
        matched_data_ids.add(data_item.id)
        description = getattr(data_item, "visual_description", "") or getattr(story_item, "visual_description", "")
        if description == getattr(data_item, "name", "") or description == getattr(story_item, "name", ""):
            description = getattr(story_item, "visual_description", "")
        merged.append(replace(data_item, visual_description=description))
    for data_item in data_items:
        if data_item.id in by_id or data_item.id in matched_data_ids or data_item.id in {"main_character", "primary_location"}:
            continue
        merged.append(data_item)
    return merged


def _dialogue_actor_names(value: str) -> list[str]:
    names = []
    for match in re.finditer(r"(?:^|\s)([A-ZÄÖÜ][A-ZÄÖÜ0-9 _'’-]{1,40}):", str(value or "")):
        name = match.group(1).strip()
        if name and name not in names:
            names.append(name)
    return names


# Backwards-compatible alias for re-export via movie_planning.py
_display_name = display_name


def _location_base_name(name: str) -> str:
    text = str(name or "").strip()
    text = re.sub(
        r"[-\u2013\u2014]\s*(DAY|NIGHT|DAWN|DUSK|MORNING|EVENING|LATER|"
        r"CONTINUOUS|MOMENTS?\s*LATER|LATE\s*(?:AFTERNOON|EVENING|MORNING)|"
        r"TIME\s*LATER|SOME\s*(?:TIME|DAYS?\b|YEARS?\b)\s*LATER)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip(" .;,")
    text = re.sub(r"\s*\([^)]*\)$", "", text).strip(" .;,")
    text = re.sub(r"^(INT\.|EXT\.|INT/EXT\.)\s*", "", text, flags=re.IGNORECASE).strip(" .;,")
    return " ".join(text.split())


def _location_id_matches(id_a: str, id_b: str, *, name_a: str, name_b: str) -> bool:
    if id_a == id_b:
        return True
    base_a = safe_id(_location_base_name(name_a))
    base_b = safe_id(_location_base_name(name_b))
    return base_a == base_b


def _location_visual_description(name: str, action: str, *, character_names: tuple[str, ...] = ()) -> str:
    clean_name = _location_base_name(name)
    visual_action = _visual_location_action(action, character_names=character_names)
    if visual_action:
        combined = f"{clean_name}. {visual_action}"
        return combined[:350]
    return clean_name


def _visual_location_action(value: str, *, character_names: tuple[str, ...] = ()) -> str:
    text = " ".join(str(value or "").split()).strip(" .;,")
    if not text:
        return ""
    text = re.sub(r"\b[A-ZÄÖÜ][A-ZÄÖÜ0-9 _'\'-]{1,40}:\s*[^.?!]+[.?!]?", "", text).strip(" .;,")
    text = re.sub(r"(?i)\b(?:says?|speaks?|asks?|answers?)\b[^.?!]*[.?!]?", "", text).strip(" .;,")
    text = re.sub(r"\s{2,}", " ", text).strip(" .;,")
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    env_sentences = []
    for sentence in sentences:
        s = sentence.strip(" .;,")
        if not s:
            continue
        if _is_character_action(s, character_names=character_names):
            continue
        env_sentences.append(s)
    if not env_sentences:
        return ""
    return ". ".join(env_sentences[:3])[:320]


def _is_character_action(sentence: str, *, character_names: tuple[str, ...]) -> bool:
    s = sentence.strip()
    s_upper = s.upper()
    if re.match(r"^(?:HE|SHE|THEY|IT)\b\s+\w", s_upper):
        return True
    if re.match(r"^(?:HIS|HER|THEIR)\b\s+\w", s_upper):
        return True
    camera_re = r"^(?:THE\s+)?(?:CAMERA\s+|WE\s+(?:SEE|CUT|PAN|TILT|ZOOM|RACK|TRACK|DOLLY)\s+|CUT\s|FADE\s|DISSOLVE\s|IRIS\s|SMASH\s+TO\s|TITLE\s|SUPER\s|GRAPHIC\s)"
    if re.match(camera_re, s_upper):
        return True
    for name in character_names:
        if not name:
            continue
        name_upper = name.upper()
        pattern = re.compile(r"^" + re.escape(name_upper) + r"\b\s+\w", re.IGNORECASE)
        if pattern.match(s):
            return True
    if re.search(r"\(V\.?O\.?\)|\(O\.?S\.?\)", s):
        return True
    return False


# Backwards-compatible alias for re-export via movie_planning.py
_clean_visual_description = clean_visual_description


def _continuity_from_data(raw) -> list[MovieContinuityRule]:
    rules = []
    for index, rule in enumerate(raw if isinstance(raw, list) else [], start=1):
        if isinstance(rule, dict):
            description = str(rule.get("description") or rule.get("rule") or "").strip()
            rule_id = safe_id(rule.get("id") or description) or f"continuity_{index}"
        else:
            description = str(rule).strip()
            rule_id = safe_id(description) or f"continuity_{index}"
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


def _split_beats(text: str) -> list[str]:
    parts = [part.strip(" .!?") for part in re.split(r"[.!?]+", text.replace("\n", " ")) if part.strip()]
    return parts[:12] or [text]


def _split_screenplay_beats(text: str) -> list[str]:
    beats = ["\n".join([scene.heading, *scene.body]) for scene in parse_screenplay(text)]
    return beats or _split_beats(" ".join(text.strip().split()))


def _parse_screenplay_beat(beat: str) -> dict[str, str] | None:
    lines = [line.strip() for line in beat.splitlines() if line.strip()]
    if not lines:
        return None
    heading = HEADING_RE.match(lines[0])
    if heading is None:
        return None

    kind = heading.group(1).upper()
    location = heading.group(2).strip()
    dialogue_text, actions = split_screenplay_dialogue(lines[1:])

    action = " ".join(actions).strip()
    camera = "controlled interior dolly with motivated cinematic framing" if kind.startswith("INT") else "wide exterior establishing move with cinematic depth"
    return {
        "location": location,
        "dialogue": dialogue_text,
        "action": action,
        "camera": camera,
        "expression": "emotion and facial acting follow the dialogue beats" if dialogue_text else "subtle emotionally grounded acting",
    }
