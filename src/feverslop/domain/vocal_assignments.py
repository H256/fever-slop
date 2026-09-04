from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import re
from typing import Any


_VOCAL_STATES = frozenset({"singing", "vocals", "vocal"})


def infer_vocal_performers(
    *,
    prompt: str,
    actors: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Preserve an LLM-authored visible vocal choice as a stable subject ID."""
    text = str(prompt or "")
    if not re.search(
        r"\b(?:sings?|singing|vocal(?:ist|s)?|lip[ -]?sync(?:ing|ed)?)\b",
        text,
        re.IGNORECASE,
    ):
        return []
    for actor in actors:
        subject_id = str(actor.get("id") or "").strip()
        name = str(actor.get("name") or "").strip()
        if subject_id and name and re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE):
            return [{"subject_id": subject_id, "speaker_id": "S1"}]
    return []


def build_generated_vocal_assignments(
    *,
    prompt_relay: Sequence[Mapping[str, Any]],
    fps: int,
    duration_seconds: float,
    performers: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized_performers = _performers(performers)
    if not normalized_performers:
        return []
    assignments = []
    for relay in prompt_relay:
        if str(relay.get("state") or "").strip().casefold() not in _VOCAL_STATES:
            continue
        start = max(0.0, float(relay.get("frame_start") or 0) / fps)
        end = min(duration_seconds, float(relay.get("frame_end") or 0) / fps)
        if end > start:
            assignments.append({
                "start_seconds": round(start, 6),
                "end_seconds": round(end, 6),
                "performers": deepcopy(normalized_performers),
            })
    return assignments


def apply_vocal_assignments(
    segment: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]],
    *,
    fps: int,
) -> dict[str, Any]:
    result = deepcopy(dict(segment))
    duration = float(result.get("duration") or result.get("duration_seconds") or 0)
    normalized = _assignments(assignments, duration=duration)
    references = result.setdefault("references", {})
    actor_ids = list(references.get("actor_ids") or [])
    bindings = []
    for assignment in normalized:
        for performer in assignment["performers"]:
            subject_id = performer["subject_id"]
            if subject_id not in actor_ids:
                actor_ids.append(subject_id)
            binding = {"stem": "vocals", **performer}
            if binding not in bindings:
                bindings.append(binding)
    references["actor_ids"] = actor_ids
    references["audio_subject_bindings"] = bindings

    relay = (result.setdefault("ltx", {})).get("prompt_relay") or []
    for item in relay:
        relay_start = float(item.get("frame_start") or 0) / fps
        relay_end = float(item.get("frame_end") or 0) / fps
        performers = []
        for assignment in normalized:
            if assignment["start_seconds"] < relay_end and assignment["end_seconds"] > relay_start:
                for performer in assignment["performers"]:
                    if performer not in performers:
                        performers.append(deepcopy(performer))
        if performers:
            item["speaker_bindings"] = performers
    return result


def _assignments(
    assignments: Sequence[Mapping[str, Any]],
    *,
    duration: float,
) -> list[dict[str, Any]]:
    normalized = []
    for assignment in assignments:
        start = float(assignment.get("start_seconds") or 0)
        end = float(assignment.get("end_seconds") or 0)
        if start < 0 or end <= start or end > duration:
            raise ValueError("vocal assignment must fit within the scene duration")
        performers = _performers(assignment.get("performers") or ())
        if not performers:
            raise ValueError("vocal assignment requires at least one performer")
        normalized.append({
            "start_seconds": start,
            "end_seconds": end,
            "performers": performers,
        })
    return normalized


def _performers(values: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    normalized = []
    seen_subjects: set[str] = set()
    seen_speakers: set[str] = set()
    for value in values:
        subject_id = str(value.get("subject_id") or "").strip()
        speaker_id = str(value.get("speaker_id") or "").strip()
        if not subject_id or not re.fullmatch(r"S[1-9][0-9]*", speaker_id):
            raise ValueError("each vocal performer requires subject_id and speaker_id S1..Sn")
        if subject_id in seen_subjects or speaker_id in seen_speakers:
            raise ValueError("vocal performer subject and speaker IDs must be unique")
        seen_subjects.add(subject_id)
        seen_speakers.add(speaker_id)
        normalized.append({"subject_id": subject_id, "speaker_id": speaker_id})
    return normalized
