"""Semantic reference-sheet planning with deterministic backend compilation."""

from __future__ import annotations

import re
from typing import Any

from feverslop.domain.reference_sheet import (
    CompiledReferenceSheetPlan,
    ReferenceSheetPlan,
)

CHARACTER_VIEWS = ("full_body", "front", "left_profile", "right_profile", "back", "closeup")
LOCATION_VIEWS = ("front", "right_side", "rear", "left_side", "wide_establishing")

_CHARACTER_ACTION = re.compile(
    r"[,;]?\s+\b(?:singing|performing|playing|drumming|dancing|running|walking|"
    r"fighting|holding|sitting|standing|posing|kneeling|jumping)\b[^,;.]*",
    re.IGNORECASE,
)
_CHARACTER_ACTION_TO_IDENTITY = re.compile(
    r"\b(?:singing|performing|playing|drumming|dancing|running|walking|fighting|"
    r"holding|sitting|standing|posing|kneeling|jumping)\b[^,;.]*?"
    r"(?=\b(?:with|wearing|featuring|having)\b)",
    re.IGNORECASE,
)
_CHARACTER_LOCATION_TAIL = re.compile(
    r"\s+\b(?:on|at|inside|outside|against|before|beside)\b[^,;.]*$",
    re.IGNORECASE,
)
_CHARACTER_IN_LOCATION_TAIL = re.compile(
    r"\s+\bin\b[^,;.]*(?:nightclub|studio|stage|altar|battlefield|kitchen|room|"
    r"hall|forest|street|city|landscape|temple|castle|cave|bar|club)[^,;.]*$",
    re.IGNORECASE,
)


def _text_list(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _identity_anchor_description(description: str) -> str:
    """Keep stable appearance text while dropping one-off action and setting tails."""
    normalized = " ".join(str(description or "").split()).strip(" .")
    identity = _CHARACTER_ACTION_TO_IDENTITY.sub("", normalized)
    identity = _CHARACTER_ACTION.sub("", identity)
    identity = _CHARACTER_LOCATION_TAIL.sub("", identity).strip(" ,;.")
    identity = _CHARACTER_IN_LOCATION_TAIL.sub("", identity).strip(" ,;.")
    return identity or normalized


def compile_reference_sheet_plan(
    plan: ReferenceSheetPlan | dict[str, Any],
    *,
    kind: str,
    description: str,
    frames: int = 124,
) -> CompiledReferenceSheetPlan:
    """Compile semantic model output into a fixed, backend-neutral contract."""
    normalized_kind = str(kind).strip().lower()
    if normalized_kind not in {"character", "location"}:
        raise ValueError("reference sheet kind must be character or location")
    source = plan if isinstance(plan, ReferenceSheetPlan) else ReferenceSheetPlan.model_validate({"kind": normalized_kind, **plan})
    if normalized_kind == "character":
        labels = CHARACTER_VIEWS
        framing = "full body, generous margin"
        coverage = "cut views"
        rotation = "none"
        backdrop = source.backdrop or "plain seamless neutral grey studio backdrop"
        anchor_description = _identity_anchor_description(
            source.anchor_description or description,
        )
    else:
        labels = LOCATION_VIEWS
        framing = "landscape"
        coverage = "cut views"
        rotation = "none"
        backdrop = source.backdrop or "the described location with no people"
        anchor_description = source.anchor_description.strip() or description.strip()
    identity = _text_list(source.identity_constraints)
    if description.strip() and description.strip() not in identity:
        identity.insert(0, description.strip())
    identity.append("clothing, hair, colors, proportions, materials, lighting and layout remain identical across views")
    negatives = _text_list(source.negative_constraints)
    negatives.append("text, watermark, split screen, contact sheet, storyboard panels and unrelated structures")
    return CompiledReferenceSheetPlan(
        kind=normalized_kind,
        view_count=len(labels),
        view_labels=labels,
        framing=framing,
        coverage=coverage,
        rotation=rotation,
        backdrop=backdrop,
        duration_seconds=max(1, int(frames)) / 24.0,
        anchor_rule="the anchor image is fully referenced as the first frame",
        identity_constraints="; ".join(dict.fromkeys(identity)),
        negative_constraints="; ".join(dict.fromkeys(negatives)),
        anchor_description=anchor_description,
    )


class DeterministicReferenceSheetPlanner:
    """Fallback planner used when DSPy is unavailable or returns bad data."""

    def plan(self, *, kind: str, description: str, asset_context: dict[str, Any]) -> ReferenceSheetPlan:
        constraints = [description.strip()]
        for key in ("wardrobe", "appearance", "materials", "fixed_traits"):
            value = asset_context.get(key)
            if value:
                constraints.append(str(value).strip())
        return ReferenceSheetPlan(
            kind=kind.strip().lower(),
            anchor_description=(
                _identity_anchor_description(description)
                if kind.strip().lower() == "character"
                else description.strip()
            ),
            view_count=6 if kind.strip().lower() == "character" else 5,
            view_labels=list(CHARACTER_VIEWS if kind.strip().lower() == "character" else LOCATION_VIEWS),
            framing="full body, generous margin" if kind.strip().lower() == "character" else "landscape",
            coverage="cut views",
            rotation="none",
            backdrop="plain seamless neutral grey studio backdrop" if kind.strip().lower() == "character" else "the described location with no people",
            duration_seconds=5.0,
            anchor_rule="the anchor image is fully referenced as the first frame",
            identity_constraints=constraints,
            negative_constraints=["text", "watermark", "split screen", "contact sheet"],
        )


class ReferenceSheetPlanner:
    """DSPy planner with a deterministic fallback at the application boundary."""

    def __init__(self, *, llm: Any | None = None, dspy_runtime: Any | None = None):
        self._fallback = DeterministicReferenceSheetPlanner()
        self._modules = None
        self.source = "deterministic"
        self.fallback_reason: str | None = None
        if llm is not None:
            try:
                from feverslop.prompting.reference_sheet_modules import (
                    ReferenceSheetPlanningModules,
                )

                self._modules = ReferenceSheetPlanningModules(llm, dspy_runtime=dspy_runtime)
            except Exception:
                self._modules = None
                self.fallback_reason = "dspy_unavailable_or_initialization_failed"

    def plan(self, *, kind: str, description: str, asset_context: dict[str, Any]) -> ReferenceSheetPlan:
        if self._modules is None:
            self.source = "deterministic_fallback"
            return self._fallback.plan(kind=kind, description=description, asset_context=asset_context)
        try:
            result = self._modules.plan(kind=kind, description=description, asset_context=asset_context)
            validated = ReferenceSheetPlan.model_validate(result)
            if (
                validated.kind.strip().lower() != kind.strip().lower()
                or validated.view_count < 1
                or not validated.view_labels
            ):
                raise ValueError("DSPy reference-sheet plan omitted required contract fields")
            self.source = "dspy"
            self.fallback_reason = None
            return validated
        except Exception as exc:
            self.source = "deterministic_fallback"
            self.fallback_reason = type(exc).__name__
            return self._fallback.plan(kind=kind, description=description, asset_context=asset_context)
    source = "deterministic"
