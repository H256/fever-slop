from __future__ import annotations

from dataclasses import dataclass

from feverslop.domain.movie_continuity import (
    MovieContinuityStyleBible,
    MovieContinuityCharacterState,
    MovieContinuityLocationState,
    MovieSceneContinuityPacket,
    MovieNarrativeBeat,
    MovieContinuityLedger,
    MovieContinuityPlan,
)
from feverslop.domain.movie_references import (
    StoryArch,
    MovieActor,
    MovieLocation,
    MovieContinuityRule,
    MovieBible,
)

__all__ = [
    "CinematicShot",
    "MovieScreenplayScene",
    "MovieScreenplayArtifact",
    "MovieAct",
    "MovieTurningPoint",
    "MovieSetupPayoff",
    "MovieCharacterArc",
    "MovieSceneBlueprint",
    "MovieStoryDesign",
    "MovieNarrativePlan",
    "MovieSceneCard",
    "MovieShotCard",
    "Screenplay",
    "MovieProject",
    "MovieContinuityStyleBible",
    "MovieContinuityCharacterState",
    "MovieContinuityLocationState",
    "MovieSceneContinuityPacket",
    "MovieNarrativeBeat",
    "MovieContinuityLedger",
    "MovieContinuityPlan",
    "StoryArch",
    "MovieActor",
    "MovieLocation",
    "MovieContinuityRule",
    "MovieBible",
]


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
    transition_from_previous: str = "cut"


@dataclass(frozen=True)
class MovieScreenplayScene:
    scene_id: str
    heading: str
    summary: str
    action: str
    dialogue: str = ""
    actor_ids: tuple[str, ...] = ()
    location_id: str = ""
    source_span: str = ""
    dramatic_purpose: str = ""
    conflict: str = ""
    emotional_turn: str = ""
    subtext: str = ""
    dialogue_function: str = ""


@dataclass(frozen=True)
class MovieScreenplayArtifact:
    title: str
    source_type: str
    dialogue_language: str
    scenes: tuple[MovieScreenplayScene, ...]


@dataclass(frozen=True)
class MovieAct:
    act_id: str
    title: str
    purpose: str
    scene_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class MovieTurningPoint:
    id: str
    scene_id: str
    description: str


@dataclass(frozen=True)
class MovieSetupPayoff:
    id: str
    setup_scene_id: str
    payoff_scene_id: str
    description: str


@dataclass(frozen=True)
class MovieCharacterArc:
    actor_id: str
    want: str = ""
    need: str = ""
    starting_state: str = ""
    ending_state: str = ""


@dataclass(frozen=True)
class MovieSceneBlueprint:
    scene_id: str
    purpose: str
    conflict: str
    emotional_turn: str
    subtext: str
    dialogue_function: str
    required_actors: tuple[str, ...] = ()
    location_id: str = ""
    expected_duration: float = 0.0


@dataclass(frozen=True)
class MovieStoryDesign:
    title: str
    premise: str
    theme: str
    act_structure: tuple[MovieAct, ...]
    turning_points: tuple[MovieTurningPoint, ...]
    setup_payoff_threads: tuple[MovieSetupPayoff, ...]
    character_arcs: tuple[MovieCharacterArc, ...]
    scene_blueprint: tuple[MovieSceneBlueprint, ...]


@dataclass(frozen=True)
class MovieNarrativePlan:
    title: str
    sequences: tuple[dict, ...]
    causal_chain: tuple[dict, ...]
    open_threads: tuple[str, ...] = ()


@dataclass(frozen=True)
class MovieSceneCard:
    scene_id: str
    shot_ids: tuple[str, ...]
    dramatic_purpose: str
    story_state_before: str
    story_state_after: str
    active_actor_ids: tuple[str, ...]
    location_id: str
    dialogue: str = ""


@dataclass(frozen=True)
class MovieShotCard:
    shot_id: str
    scene_id: str
    action: str
    camera: str
    acting: str
    dialogue: str = ""
    start_frame_brief: str = ""
    end_frame_brief: str = ""
    transition_from_previous: str = "cut"
    transition_reason: str = ""


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
