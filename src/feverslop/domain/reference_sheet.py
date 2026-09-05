"""Backend-neutral contracts for turning generated sequences into reference sheets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BaseModel, Field, model_validator


@dataclass(frozen=True, slots=True)
class CompiledReferenceSheetPlan:
    kind: str
    view_count: int
    view_labels: tuple[str, ...]
    framing: str
    coverage: str
    rotation: str
    backdrop: str
    duration_seconds: float
    anchor_rule: str
    identity_constraints: str
    negative_constraints: str
    anchor_description: str = ""

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "view_count": self.view_count,
            "view_labels": list(self.view_labels),
            "framing": self.framing,
            "coverage": self.coverage,
            "rotation": self.rotation,
            "backdrop": self.backdrop,
            "duration_seconds": self.duration_seconds,
            "anchor_rule": self.anchor_rule,
            "identity_constraints": self.identity_constraints,
            "negative_constraints": self.negative_constraints,
            "anchor_description": self.anchor_description,
        }


class ReferenceSheetPlan(BaseModel):
    kind: str = Field(description="Either character or location.")
    anchor_description: str = Field(default="", max_length=2000, description="Stable anchor identity or environment description.")
    view_count: int = Field(default=0, ge=0, le=12, description="Number of views represented by view_labels.")
    view_labels: list[str] = Field(default_factory=list, max_length=12, description="One short label per requested view, in render order.")
    framing: str = Field(default="", max_length=256)
    coverage: str = Field(default="", max_length=256)
    rotation: str = Field(default="", max_length=256)
    backdrop: str = Field(default="", max_length=512)
    duration_seconds: float = Field(default=0.0, ge=0.0, le=600.0)
    anchor_rule: str = Field(default="", max_length=512)
    identity_constraints: list[str] = Field(default_factory=list, max_length=32)
    negative_constraints: list[str] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def validate_contract(self) -> "ReferenceSheetPlan":
        if self.kind.strip().lower() not in {"character", "location"}:
            raise ValueError("kind must be character or location")
        if self.view_count == 0 and not self.view_labels:
            return self
        if len(self.view_labels) != self.view_count:
            raise ValueError("view_count must equal len(view_labels)")
        if any(not isinstance(label, str) or not 1 <= len(label.strip()) <= 64 for label in self.view_labels):
            raise ValueError("view_labels must contain non-empty labels of at most 64 characters")
        return self


class ReferenceSheetPlanContract(BaseModel):
    """Strict model-output contract used by the DSPy signature."""

    kind: str = Field(description="Either character or location.")
    anchor_description: str = Field(max_length=2000)
    view_count: int = Field(ge=1, le=12, description="Must equal the number of entries in view_labels.")
    view_labels: list[Annotated[str, Field(min_length=1, max_length=64)]] = Field(
        min_length=1,
        max_length=12,
        description="1-12 labels; each label is 1-64 characters and view_count must match the list length.",
    )
    framing: str = Field(max_length=256)
    coverage: str = Field(max_length=256)
    rotation: str = Field(max_length=256)
    backdrop: str = Field(max_length=512)
    duration_seconds: float = Field(ge=0.0, le=600.0)
    anchor_rule: str = Field(max_length=512)
    identity_constraints: list[str] = Field(max_length=32)
    negative_constraints: list[str] = Field(max_length=32)

    @model_validator(mode="after")
    def validate_contract(self) -> "ReferenceSheetPlanContract":
        if self.kind.strip().lower() not in {"character", "location"}:
            raise ValueError("kind must be character or location")
        if len(self.view_labels) != self.view_count:
            raise ValueError("view_count must equal len(view_labels)")
        if any(not label.strip() or len(label.strip()) > 64 for label in self.view_labels):
            raise ValueError("view_labels must contain labels of at most 64 characters")
        return self

