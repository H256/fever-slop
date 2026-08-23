from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from feverslop.domain.movie import (
    CinematicShot,
    MovieActor,
    MovieBible,
    MovieLocation,
    MovieScreenplayArtifact,
    StoryArch,
)


class MoviePlanningResult(BaseModel):
    model_config = ConfigDict(extra="allow")


class MoviePlanningPayload(BaseModel):
    model_config = ConfigDict(extra="allow")


class StoryArchPayload(MoviePlanningPayload):
    title: str = ""
    source_type: str = ""
    story_text: str = ""
    desired_length: float = 0.0


class MovieBiblePayload(StoryArchPayload):
    story_arch: StoryArch | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class RefineLocationsPayload(MoviePlanningPayload):
    locations: tuple[MovieLocation, ...] = ()
    source_text: str = ""
    guide: str = ""


class RefineActorsPayload(MoviePlanningPayload):
    actors: tuple[MovieActor, ...] = ()
    source_text: str = ""
    premise: str = ""
    guide: str = ""


class ContinuityPlanPayload(MoviePlanningPayload):
    title: str = ""
    source_type: str = ""
    story_text: str = ""
    desired_length: float = 0.0
    bible: MovieBible | None = None
    shots: tuple[CinematicShot, ...] = ()
    config: dict[str, Any] = Field(default_factory=dict)


class StoryDesignPayload(MoviePlanningPayload):
    title: str = ""
    source_type: str = ""
    story_text: str = ""
    desired_length: float = 0.0
    bible: MovieBible | None = None
    story_arch: StoryArch | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class ScreenplayPayload(StoryDesignPayload):
    story_design: Any = None


class NarrativePlanPayload(MoviePlanningPayload):
    title: str = ""
    source_type: str = ""
    desired_length: float = 0.0
    bible: MovieBible | None = None
    screenplay: MovieScreenplayArtifact | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class ShotPlanFromBiblePayload(MoviePlanningPayload):
    bible: MovieBible | None = None
    screenplay: MovieScreenplayArtifact | None = None
    desired_length: float = 0.0
    width: int = 0
    height: int = 0
    min_duration: float = 4.0
    max_duration: float = 20.0


class ShotPlanPayload(MoviePlanningPayload):
    story_arch: StoryArch | None = None
    desired_length: float = 0.0
    width: int = 0
    height: int = 0
    min_duration: float = 4.0
    max_duration: float = 20.0


class MovieActorResult(MoviePlanningResult):
    id: str = ""
    name: str = ""
    role: str = ""
    visual_description: str = ""


class MovieLocationResult(MoviePlanningResult):
    id: str = ""
    name: str = ""
    visual_description: str = ""
    image_prompt: str = ""


class MovieContinuityRuleResult(MoviePlanningResult):
    id: str = ""
    description: str = ""


class MovieShotResult(MoviePlanningResult):
    shot_id: str = ""
    description: str = ""
    duration_seconds: float = 0.0
    camera: str = ""
    action: str = ""
    expression: str = ""
    location: str = ""
    dialogue: str = ""
    actor_ids: list[str] = Field(default_factory=list)
    location_id: str = ""
    transition_from_previous: str = "cut"


class MovieActResult(MoviePlanningResult):
    act_id: str = ""
    title: str = ""
    purpose: str = ""
    scene_ids: list[str] = Field(default_factory=list)


class MovieTurningPointResult(MoviePlanningResult):
    id: str = ""
    scene_id: str = ""
    description: str = ""


class MovieSetupPayoffResult(MoviePlanningResult):
    id: str = ""
    setup_scene_id: str = ""
    payoff_scene_id: str = ""
    description: str = ""


class MovieCharacterArcResult(MoviePlanningResult):
    actor_id: str = ""
    want: str = ""
    need: str = ""
    starting_state: str = ""
    ending_state: str = ""


class MovieSceneBlueprintResult(MoviePlanningResult):
    scene_id: str = ""
    purpose: str = ""
    conflict: str = ""
    emotional_turn: str = ""
    subtext: str = ""
    dialogue_function: str = ""
    required_actors: list[str] = Field(default_factory=list)
    location_id: str = ""
    expected_duration: float = 0.0


class MovieScreenplaySceneResult(MoviePlanningResult):
    scene_id: str = ""
    heading: str = ""
    summary: str = ""
    action: str = ""
    dialogue: str = ""
    actor_ids: list[str] = Field(default_factory=list)
    location_id: str = ""
    source_span: str = ""
    dramatic_purpose: str = ""
    conflict: str = ""
    emotional_turn: str = ""
    subtext: str = ""
    dialogue_function: str = ""


class MovieNarrativeItemResult(MoviePlanningResult):
    scene_ids: list[str] = Field(default_factory=list)
    description: str = ""
    cause_from_previous: str = ""
    sets_up_next: str = ""


class StoryArchResult(MoviePlanningResult):
    title: str = ""
    premise: str = ""
    beats: list[str] = Field(default_factory=list)


class MovieBibleResult(MoviePlanningResult):
    title: str = ""
    premise: str = ""
    actors: list[MovieActorResult] = Field(default_factory=list)
    locations: list[MovieLocationResult] = Field(default_factory=list)
    continuity: list[MovieContinuityRuleResult] = Field(default_factory=list)
    style_constraints: list[str] = Field(default_factory=list)


class RefinedLocationsResult(MoviePlanningResult):
    locations: list[MovieLocationResult] = Field(default_factory=list)


class RefinedActorsResult(MoviePlanningResult):
    actors: list[MovieActorResult] = Field(default_factory=list)


class ContinuityPlanResult(MoviePlanningResult):
    continuity_ledger: dict[str, Any] = Field(default_factory=dict)
    scene_continuity: dict[str, Any] = Field(default_factory=dict)
    narrative_chain: list[MovieNarrativeItemResult] = Field(default_factory=list)


class StoryDesignResult(MoviePlanningResult):
    title: str = ""
    premise: str = ""
    theme: str = ""
    act_structure: list[MovieActResult] = Field(default_factory=list)
    turning_points: list[MovieTurningPointResult] = Field(default_factory=list)
    setup_payoff_threads: list[MovieSetupPayoffResult] = Field(default_factory=list)
    character_arcs: list[MovieCharacterArcResult] = Field(default_factory=list)
    scene_blueprint: list[MovieSceneBlueprintResult] = Field(default_factory=list)


class ScreenplayResult(MoviePlanningResult):
    title: str = ""
    source_type: str = ""
    dialogue_language: str = ""
    scenes: list[MovieScreenplaySceneResult] = Field(default_factory=list)


class NarrativePlanResult(MoviePlanningResult):
    title: str = ""
    sequences: list[MovieNarrativeItemResult] = Field(default_factory=list)
    causal_chain: list[MovieNarrativeItemResult] = Field(default_factory=list)
    open_threads: list[str] = Field(default_factory=list)


class ShotPlanResult(MoviePlanningResult):
    shots: list[MovieShotResult] = Field(default_factory=list)


def build_movie_planning_signature_bundle(dspy_module: Any | None = None) -> dict[str, Any]:
    if dspy_module is None:
        import dspy as dspy_module

    class StoryArch(dspy_module.Signature):
        """Create a movie story arch from the supplied structured source data."""

        guide: str = dspy_module.InputField()
        payload: StoryArchPayload = dspy_module.InputField()
        result: StoryArchResult = dspy_module.OutputField()

    class MovieBible(dspy_module.Signature):
        """Create a typed movie bible from the supplied structured source data."""

        guide: str = dspy_module.InputField()
        payload: MovieBiblePayload = dspy_module.InputField()
        result: MovieBibleResult = dspy_module.OutputField()

    class RefineLocations(dspy_module.Signature):
        """Refine location descriptions while preserving location ids and order."""

        guide: str = dspy_module.InputField()
        payload: RefineLocationsPayload = dspy_module.InputField()
        result: RefinedLocationsResult = dspy_module.OutputField()

    class RefineActors(dspy_module.Signature):
        """Refine actor visual descriptions while preserving actor ids and order."""

        guide: str = dspy_module.InputField()
        payload: RefineActorsPayload = dspy_module.InputField()
        result: RefinedActorsResult = dspy_module.OutputField()

    class ContinuityPlan(dspy_module.Signature):
        """Build a typed continuity and narrative plan for movie shots."""

        guide: str = dspy_module.InputField()
        payload: ContinuityPlanPayload = dspy_module.InputField()
        result: ContinuityPlanResult = dspy_module.OutputField()

    class StoryDesign(dspy_module.Signature):
        """Create a typed dramaturgical story design before screenplay writing."""

        guide: str = dspy_module.InputField()
        payload: StoryDesignPayload = dspy_module.InputField()
        result: StoryDesignResult = dspy_module.OutputField()

    class Screenplay(dspy_module.Signature):
        """Create the canonical typed screenplay while preserving source ordering."""

        guide: str = dspy_module.InputField()
        payload: ScreenplayPayload = dspy_module.InputField()
        result: ScreenplayResult = dspy_module.OutputField()

    class NarrativePlan(dspy_module.Signature):
        """Create typed narrative memory from the canonical screenplay."""

        guide: str = dspy_module.InputField()
        payload: NarrativePlanPayload = dspy_module.InputField()
        result: NarrativePlanResult = dspy_module.OutputField()

    class ShotPlanFromBible(dspy_module.Signature):
        """Create a typed cinematic shot plan from a movie bible and screenplay."""

        guide: str = dspy_module.InputField()
        payload: ShotPlanFromBiblePayload = dspy_module.InputField()
        result: ShotPlanResult = dspy_module.OutputField()

    class ShotPlan(dspy_module.Signature):
        """Create a typed cinematic shot plan from a story arch."""

        guide: str = dspy_module.InputField()
        payload: ShotPlanPayload = dspy_module.InputField()
        result: ShotPlanResult = dspy_module.OutputField()

    return {
        "story_arch": StoryArch,
        "movie_bible": MovieBible,
        "refine_locations": RefineLocations,
        "refine_actors": RefineActors,
        "continuity_plan": ContinuityPlan,
        "story_design": StoryDesign,
        "screenplay": Screenplay,
        "narrative_plan": NarrativePlan,
        "shot_plan_from_bible": ShotPlanFromBible,
        "shot_plan": ShotPlan,
    }
