from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MoviePlanningResult(BaseModel):
    model_config = ConfigDict(extra="allow")


class StoryArchResult(MoviePlanningResult):
    title: str = ""
    premise: str = ""
    beats: list[str] = Field(default_factory=list)


class MovieBibleResult(MoviePlanningResult):
    title: str = ""
    premise: str = ""
    actors: list[dict[str, Any]] = Field(default_factory=list)
    locations: list[dict[str, Any]] = Field(default_factory=list)
    continuity: list[dict[str, Any]] = Field(default_factory=list)
    style_constraints: list[str] = Field(default_factory=list)


class RefinedLocationsResult(MoviePlanningResult):
    locations: list[dict[str, Any]] = Field(default_factory=list)


class RefinedActorsResult(MoviePlanningResult):
    actors: list[dict[str, Any]] = Field(default_factory=list)


class ContinuityPlanResult(MoviePlanningResult):
    continuity_ledger: dict[str, Any] = Field(default_factory=dict)
    scene_continuity: dict[str, Any] = Field(default_factory=dict)
    narrative_chain: list[dict[str, Any]] = Field(default_factory=list)


class StoryDesignResult(MoviePlanningResult):
    title: str = ""
    premise: str = ""
    theme: str = ""
    act_structure: list[dict[str, Any]] = Field(default_factory=list)
    turning_points: list[dict[str, Any]] = Field(default_factory=list)
    setup_payoff_threads: list[dict[str, Any]] = Field(default_factory=list)
    character_arcs: list[dict[str, Any]] = Field(default_factory=list)
    scene_blueprint: list[dict[str, Any]] = Field(default_factory=list)


class ScreenplayResult(MoviePlanningResult):
    title: str = ""
    source_type: str = ""
    dialogue_language: str = ""
    scenes: list[dict[str, Any]] = Field(default_factory=list)


class NarrativePlanResult(MoviePlanningResult):
    title: str = ""
    sequences: list[dict[str, Any]] = Field(default_factory=list)
    causal_chain: list[dict[str, Any]] = Field(default_factory=list)
    open_threads: list[str] = Field(default_factory=list)


class ShotPlanResult(MoviePlanningResult):
    shots: list[dict[str, Any]] = Field(default_factory=list)


def build_movie_planning_signature_bundle(dspy_module: Any | None = None) -> dict[str, Any]:
    if dspy_module is None:
        import dspy as dspy_module

    class StoryArch(dspy_module.Signature):
        """Create a movie story arch from the supplied structured source data."""
        guide: str = dspy_module.InputField()
        payload: dict[str, Any] = dspy_module.InputField()
        result: StoryArchResult = dspy_module.OutputField()

    class MovieBible(dspy_module.Signature):
        """Create a typed movie bible from the supplied structured source data."""
        guide: str = dspy_module.InputField()
        payload: dict[str, Any] = dspy_module.InputField()
        result: MovieBibleResult = dspy_module.OutputField()

    class RefineLocations(dspy_module.Signature):
        """Refine location descriptions while preserving location ids and order."""
        guide: str = dspy_module.InputField()
        payload: dict[str, Any] = dspy_module.InputField()
        result: RefinedLocationsResult = dspy_module.OutputField()

    class RefineActors(dspy_module.Signature):
        """Refine actor visual descriptions while preserving actor ids and order."""
        guide: str = dspy_module.InputField()
        payload: dict[str, Any] = dspy_module.InputField()
        result: RefinedActorsResult = dspy_module.OutputField()

    class ContinuityPlan(dspy_module.Signature):
        """Build a typed continuity and narrative plan for movie shots."""
        guide: str = dspy_module.InputField()
        payload: dict[str, Any] = dspy_module.InputField()
        result: ContinuityPlanResult = dspy_module.OutputField()

    class StoryDesign(dspy_module.Signature):
        """Create a typed dramaturgical story design before screenplay writing."""
        guide: str = dspy_module.InputField()
        payload: dict[str, Any] = dspy_module.InputField()
        result: StoryDesignResult = dspy_module.OutputField()

    class Screenplay(dspy_module.Signature):
        """Create the canonical typed screenplay while preserving source ordering."""
        guide: str = dspy_module.InputField()
        payload: dict[str, Any] = dspy_module.InputField()
        result: ScreenplayResult = dspy_module.OutputField()

    class NarrativePlan(dspy_module.Signature):
        """Create typed narrative memory from the canonical screenplay."""
        guide: str = dspy_module.InputField()
        payload: dict[str, Any] = dspy_module.InputField()
        result: NarrativePlanResult = dspy_module.OutputField()

    class ShotPlanFromBible(dspy_module.Signature):
        """Create a typed cinematic shot plan from a movie bible and screenplay."""
        guide: str = dspy_module.InputField()
        payload: dict[str, Any] = dspy_module.InputField()
        result: ShotPlanResult = dspy_module.OutputField()

    class ShotPlan(dspy_module.Signature):
        """Create a typed cinematic shot plan from a story arch."""
        guide: str = dspy_module.InputField()
        payload: dict[str, Any] = dspy_module.InputField()
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
