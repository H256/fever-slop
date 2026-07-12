from __future__ import annotations

from dataclasses import dataclass


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
