from __future__ import annotations

import re
from dataclasses import replace
from math import ceil

from feverslop.domain.movie import CinematicShot
from feverslop.domain.movie_utils import safe_id, transition_from_previous


def _beat_text(beat) -> str:
    if isinstance(beat, dict):
        return str(beat.get("summary") or beat.get("description") or beat.get("beat") or "").strip()
    return str(beat).strip()


def _safe_id(value: object) -> str:
    return safe_id(value)


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


def _normalize_movie_shots(shots, *, desired_length: float, min_duration: float, max_duration: float):
    if not shots:
        return tuple(shots)
    min_duration = max(1.0, float(min_duration))
    max_duration = max(min_duration, float(max_duration))
    expanded = []
    for shot in shots:
        duration = max(1.0, float(shot.duration_seconds))
        parts = max(1, ceil(duration / max_duration))
        for part in range(parts):
            description = _shot_part_text(shot.description, part=part, parts=parts) if parts > 1 else shot.description
            action = _shot_part_text(shot.action, part=part, parts=parts) if parts > 1 else shot.action
            if parts > 1 and part > 0 and action == shot.action:
                action = f"Continue the same scene beat: {action}"
            expanded.append(
                replace(
                    shot,
                    shot_id=f"{shot.shot_id}_{part + 1}" if parts > 1 else shot.shot_id,
                    duration_seconds=duration / parts,
                    description=description,
                    action=action,
                    dialogue=shot.dialogue if part == 0 else "",
                    expression=shot.expression if part == 0 or not shot.dialogue else "silent physical reaction after the dialogue beat",
                ),
            )
    pattern = (0.86, 1.08, 0.94, 1.18, 1.0, 0.78, 1.12)
    weighted = [max(min_duration, min(max_duration, shot.duration_seconds * pattern[index % len(pattern)])) for index, shot in enumerate(expanded)]
    total = sum(weighted) or 1.0
    scaled = [max(min_duration, min(max_duration, value * float(desired_length) / total)) for value in weighted]
    return tuple(replace(shot, shot_id=f"shot_{index:04}", duration_seconds=round(duration, 3)) for index, (shot, duration) in enumerate(zip(expanded, scaled), start=1))


def _shot_part_text(value: str, *, part: int, parts: int) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]
    if len(sentences) >= parts:
        index = min(part, len(sentences) - 1)
        return sentences[index]
    if sentences:
        if part < len(sentences):
            return sentences[part]
        return f"Continue visually from the previous moment: {sentences[-1]}"
    if part == 0:
        return text
    return f"Continue visually from the previous moment: {text}"


def _ensure_minimum_actors(shots, story_arch):
    minimum = _minimum_actor_count(story_arch)
    if minimum <= 0 or not shots:
        return shots
    actor_ids = []
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


def _minimum_actor_count(story_arch) -> int:
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


def _transition_from_previous(value: object) -> str:
    return transition_from_previous(value)


def _shots_from_data(shots, *, desired_length: float, min_duration: float, max_duration: float):
    planned = []
    for index, raw_shot in enumerate(shots, start=1):
        shot = raw_shot if isinstance(raw_shot, dict) else {"description": str(raw_shot)}
        planned.append(
            CinematicShot(
                shot_id=str(shot.get("shot_id") or f"shot_{index:04}"),
                description=str(shot.get("description") or shot.get("action") or f"Shot {index}").strip(),
                duration_seconds=float(shot.get("duration_seconds") or max(1.0, desired_length / len(shots))),
                camera=str(shot.get("camera") or "motivated cinematic camera movement").strip(),
                action=str(shot.get("action") or shot.get("description") or "").strip(),
                expression=str(shot.get("acting") or shot.get("expression") or "emotionally grounded performance").strip(),
                location=str(shot.get("location") or "story-consistent cinematic location").strip(),
                dialogue=str(shot.get("dialogue") or "").strip(),
                actor_ids=tuple(_string_list(shot.get("actor_ids") or shot.get("actors"))),
                location_id=_safe_id(shot.get("location_id") or shot.get("location")),
                transition_from_previous=_transition_from_previous(shot.get("transition_from_previous")),
            ),
        )
    return _normalize_movie_shots(
        planned,
        desired_length=float(desired_length),
        min_duration=float(min_duration),
        max_duration=float(max_duration),
    )
