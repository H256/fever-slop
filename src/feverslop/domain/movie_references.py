from __future__ import annotations

from dataclasses import dataclass


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
    image_prompt: str = ""


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
