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
    continuity_notes: str = ""
    story_state_before: str = ""
    story_state_after: str = ""
    cause_from_previous: str = ""
    narrative_purpose: str = ""
    conflict_or_tension: str = ""
    turning_point: str = ""
    sets_up_next: str = ""


@dataclass(frozen=True)
class MovieContinuityStyleBible:
    visual_style: str = ""
    palette: str = ""
    lighting: str = ""
    camera: str = ""
    negative_constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class MovieContinuityCharacterState:
    character_id: str
    base_identity: str = ""
    wardrobe: str = ""
    carried_props: tuple[str, ...] = ()
    physical_state: str = ""
    emotional_state: str = ""
    last_location: str = ""
    last_action: str = ""


@dataclass(frozen=True)
class MovieContinuityLocationState:
    location_id: str
    name: str = ""
    time_of_day: str = ""
    lighting: str = ""
    props: tuple[str, ...] = ()
    environmental_state: str = ""


@dataclass(frozen=True)
class MovieSceneContinuityPacket:
    shot_id: str
    location_id: str = ""
    incoming: tuple[str, ...] = ()
    required_carryovers: tuple[str, ...] = ()
    allowed_changes: tuple[str, ...] = ()
    outgoing: tuple[str, ...] = ()
    characters: dict[str, MovieContinuityCharacterState] | None = None
    location: MovieContinuityLocationState | None = None


@dataclass(frozen=True)
class MovieNarrativeBeat:
    shot_id: str
    story_state_before: str = ""
    story_state_after: str = ""
    cause_from_previous: str = ""
    narrative_purpose: str = ""
    conflict_or_tension: str = ""
    turning_point: str = ""
    sets_up_next: str = ""


@dataclass(frozen=True)
class MovieContinuityLedger:
    style_bible: MovieContinuityStyleBible
    characters: dict[str, MovieContinuityCharacterState]
    locations: dict[str, MovieContinuityLocationState]
    scene_order: tuple[str, ...]


@dataclass(frozen=True)
class MovieContinuityPlan:
    continuity_ledger: MovieContinuityLedger
    scene_continuity: dict[str, MovieSceneContinuityPacket]
    narrative_chain: tuple[MovieNarrativeBeat, ...]


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
