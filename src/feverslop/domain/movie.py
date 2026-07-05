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
