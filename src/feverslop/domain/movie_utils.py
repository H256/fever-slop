"""Shared utility functions for movie pipeline processing.

Centralizes utilities that were previously duplicated across the application
and adapter layers.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# safe_id / string_list
# ---------------------------------------------------------------------------


def safe_id(value: Any, fallback: str = "") -> str:
    """Create a safe identifier from an arbitrary value.

    Normalizes to lowercase alphanumeric + underscores. Returns *fallback*
    when the cleaned value is empty (only when a fallback is provided).
    """
    raw = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return raw or fallback


def string_list(value: Any) -> list[str]:
    """Convert a value to a list of non-empty stripped strings.

    Handles lists, tuples, and single strings.
    """
    if isinstance(value, list | tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def safe_id_list(value: Any) -> list[str]:
    """Convert a value to a list of safe identifiers.

    Like :func:`string_list`, but each item is passed through :func:`safe_id`.
    """
    if isinstance(value, list):
        return [safe_id(item) for item in value if safe_id(item)]
    if isinstance(value, str) and value.strip():
        return [safe_id(value)]
    return []


# ---------------------------------------------------------------------------
# transition
# ---------------------------------------------------------------------------


def transition_from_previous(value: Any) -> str:
    """Normalize a transition value to ``'continuous'`` or ``'cut'``."""
    transition = str(value or "cut").strip().lower().replace("_", "-")
    return "continuous" if transition == "continuous" else "cut"


# ---------------------------------------------------------------------------
# display name
# ---------------------------------------------------------------------------


def display_name(value: str) -> str:
    """Normalize a display name, title-casing if the input is all uppercase."""
    text = " ".join(str(value or "").split()).strip()
    return text.title() if text.isupper() else text


# ---------------------------------------------------------------------------
# visual description cleaning
# ---------------------------------------------------------------------------

_GENERIC_VISUAL_TOKENS = (
    "story-defined cinematic",
    "consistent face",
    "consistent production design",
    "body shape",
    "wardrobe, and posture",
    "consistent production",
    "geography, lighting, and atmosphere",
    "lighting, geography, and atmosphere",
)


def clean_visual_description(value: Any, fallback: str) -> str:
    """Clean a visual description, returning *fallback* for generic content.

    Strips whitespace, removes trailing punctuation, and checks against a
    combined set of known-generic token patterns (actors + locations).
    """
    text = " ".join(str(value or "").split()).strip(" .;,")
    fallback_text = " ".join(str(fallback or "").split()).strip() or "Reference"
    if not text:
        return fallback_text
    lower = text.lower()
    if lower.endswith(" with") or any(token in lower for token in _GENERIC_VISUAL_TOKENS):
        return fallback_text
    return text


# ---------------------------------------------------------------------------
# configured actors / locations
# ---------------------------------------------------------------------------


def configured_actors(config: dict) -> list:
    """Build actor list from project config, cleaning visual descriptions."""
    from feverslop.domain.movie_references import MovieActor

    actors = []
    raw = config.get("actors") if isinstance(config.get("actors"), list) else []
    for index, actor in enumerate(raw, start=1):
        if not isinstance(actor, dict):
            continue
        name = str(actor.get("name") or actor.get("id") or f"Actor {index}").strip()
        actors.append(
            MovieActor(
                id=safe_id(actor.get("id") or actor.get("name"), f"actor_{index}"),
                name=name,
                role=str(actor.get("role") or "").strip(),
                visual_description=clean_visual_description(
                    actor.get("visual_description") or actor.get("image_prompt") or actor.get("prompt"),
                    name,
                ),
            ),
        )
    return actors


def configured_locations(config: dict) -> list:
    """Build location list from project config, cleaning visual descriptions."""
    from feverslop.domain.movie_references import MovieLocation

    raw = config.get("structured_locations")
    if not isinstance(raw, list) or not raw:
        raw = config.get("locations") if isinstance(config.get("locations"), list) else []
    locations = []
    for index, location in enumerate(raw, start=1):
        if isinstance(location, dict):
            name = str(location.get("name") or location.get("id") or f"Location {index}").strip()
            locations.append(
                MovieLocation(
                    id=safe_id(location.get("id") or location.get("name"), f"location_{index}"),
                    name=name,
                    visual_description=clean_visual_description(
                        location.get("visual_description") or location.get("image_prompt") or location.get("prompt"),
                        name,
                    ),
                ),
            )
        elif str(location or "").strip():
            name = str(location).strip()
            locations.append(MovieLocation(id=safe_id(name, f"location_{index}"), name=name, visual_description=name))
    return locations


def movie_slug(plan: dict[str, Any], project_dir: Path) -> str:
    title = str(plan.get("title") or "").strip()
    if not title:
        return project_dir.name
    return "".join(char.lower() if char.isalnum() else "-" for char in title).strip("-") or project_dir.name
