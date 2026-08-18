"""Pure policies for choosing music-performance reference stems."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any


def _performance_text(segment: Mapping[str, Any]) -> str:
    references = segment.get("references") or {}
    actors = references.get("actor_reference_descriptions") or []
    actor_text = " ".join(
        " ".join(str(item.get(key) or "") for key in ("name", "role", "visual_description", "image_prompt"))
        for item in actors
        if isinstance(item, Mapping)
    )
    metadata = segment.get("metadata") or {}
    return " ".join((
        actor_text,
        str(metadata.get("base_concept") or ""),
        str(metadata.get("character_motion") or ""),
    )).casefold()


def is_fully_instrumental(segment: Mapping[str, Any]) -> bool:
    relay = (segment.get("ltx") or {}).get("prompt_relay") or segment.get("prompt_relay") or []
    if relay:
        return all(
            str(item.get("state") or "").strip().lower() == "instrumental"
            for item in relay
        )
    return str(segment.get("type") or "").strip().lower() == "instrumental"


def visible_performance_roles(segment: Mapping[str, Any]) -> set[str]:
    text = _performance_text(segment)
    roles = set()
    if any(token in text for token in ("drummer", "drumming", "percussion")):
        roles.add("drums")
    if any(token in text for token in ("bassist", "bass guitar", "plays bass")):
        roles.add("bass")
    if any(token in text for token in ("guitarist", "lead guitar", "electric guitar")):
        roles.add("other")
    if any(token in text for token in ("singer", "vocalist", "lead vocal")):
        roles.add("vocals")
    return roles


def select_performance_stems(
    segment: Mapping[str, Any],
    *,
    available_stems: Collection[str],
    max_stems: int = 2,
) -> list[str]:
    """Choose one visible performer stem plus the full mix deterministically."""
    available = set(available_stems)
    roles = visible_performance_roles(segment)
    selected: list[str] = []
    if not is_fully_instrumental(segment) and "vocals" in roles and "vocals" in available:
        selected.append("vocals")
    else:
        for stem in ("drums", "bass", "other"):
            if stem in roles and stem in available:
                selected.append(stem)
                break
    if not selected and not is_fully_instrumental(segment) and str(segment.get("type") or "").lower() == "vocals":
        if "vocals" in available:
            selected.append("vocals")
    if "full_mix" in available:
        selected.append("full_mix")
    if not selected:
        selected.extend(name for name in ("drums", "bass", "other", "vocals") if name in available)
    return list(dict.fromkeys(selected))[:max(1, int(max_stems))]


def select_performance_audio_paths(
    segment: Mapping[str, Any],
    available: Mapping[str, Path],
    *,
    max_stems: int = 2,
) -> dict[str, Path]:
    names = select_performance_stems(
        segment,
        available_stems=available.keys(),
        max_stems=max_stems,
    )
    return {name: available[name] for name in names}
