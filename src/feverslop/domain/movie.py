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
class MovieActor:
    id: str
    name: str
    role: str = ""
    visual_description: str = ""


@dataclass(frozen=True)
class MovieLocation:
    id: str
    name: str
    visual_description: str = ""


@dataclass(frozen=True)
class MovieContinuityRule:
    id: str
    description: str


@dataclass(frozen=True)
class MovieBible:
    title: str
    premise: str
    story_arch: StoryArch
    actors: tuple[MovieActor, ...]
    locations: tuple[MovieLocation, ...]
    continuity: tuple[MovieContinuityRule, ...]
    style_constraints: tuple[str, ...]
    runtime_constraints: dict


@dataclass(frozen=True)
class Screenplay:
    text: str


@dataclass(frozen=True)
class MovieProject:
    slug: str
    name: str
    bible: MovieBible
    story_arch: StoryArch
    shots: tuple[CinematicShot, ...]
    duration_seconds: float
    width: int
    height: int
    mode: str
    config: dict | None = None
