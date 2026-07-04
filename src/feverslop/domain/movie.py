from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CinematicShot:
    shot_id: str
    description: str
    duration_seconds: float
    camera: str
    action: str
    expression: str
    location: str
    dialogue: str = ""
    actor_ids: tuple[str, ...] = ()
    location_id: str = ""


@dataclass(frozen=True)
class StoryArch:
    title: str
    premise: str
    beats: tuple[str, ...]


@dataclass(frozen=True)
class Screenplay:
    text: str


@dataclass(frozen=True)
class MovieProject:
    slug: str
    name: str
    story_arch: StoryArch
    shots: tuple[CinematicShot, ...]
    duration_seconds: float
    width: int
    height: int
    mode: str
    config: dict | None = None
