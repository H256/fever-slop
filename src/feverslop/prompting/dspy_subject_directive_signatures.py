from __future__ import annotations

from typing import Literal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DirectiveTemporalScope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_seconds: float
    end_seconds: float


class DirectivePropBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prop_id: str
    state: Literal["held", "played", "attached", "placed", "absent"]
    detail: str = ""


class DirectiveSubject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject_id: str
    role: str
    position: str
    action: str
    interactions: list[str] = Field(default_factory=list)
    gaze_direction: str = ""
    prop_bindings: list[DirectivePropBinding] = Field(default_factory=list)
    visibility: str = "visible"
    cardinality: int = 1
    temporal_scope: DirectiveTemporalScope


class DirectiveSpatialRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject_id: str
    relation: str
    target_id: str
    detail: str = ""


class DirectiveStagingPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "subject-directives/v1"
    shot_id: str
    temporal_scope: DirectiveTemporalScope
    subjects: list[DirectiveSubject] = Field(default_factory=list)
    spatial_relations: list[DirectiveSpatialRelation] = Field(default_factory=list)


def build_subject_directive_signature(dspy_module: Any | None = None) -> Any:
    """Return the shared staging signature used by optional DSPy planners."""
    if dspy_module is None:
        import dspy as dspy_module

    class BuildSharedStagingPlan(dspy_module.Signature):
        """Resolve one coherent shot space before generating subject prose.

        The output must retain every visible subject, action, position, relation,
        prop binding/state, visibility/cardinality, and temporal scope from the input.
        Use the supplied scene duration as the complete shot scope: start_seconds=0
        and end_seconds=duration_seconds, with end_seconds strictly greater than
        start_seconds. Every subject must have a valid temporal_scope. If
        repair_feedback is present, correct the previous validation failure.
        Return only the staging_plan object. Do not echo the input, guide, or
        DSPy formatting markers. Do not make independent subject decisions outside
        the shared staging plan.
        """

        scene: dict[str, Any] = dspy_module.InputField()
        staging_plan: DirectiveStagingPlan = dspy_module.OutputField()

    return BuildSharedStagingPlan


__all__ = ["DirectiveStagingPlan", "build_subject_directive_signature"]
