from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from feverslop.ports.llm import VisionLLMPort
from feverslop.domain.vision_references import ReferenceImage
from feverslop.errors import FeverSlopLMLError
from feverslop.prompting.ingredients_modules import IngredientsPromptModules

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngredientsPromptResult:
    reference_description: str
    shot_invariants: str
    fallback_reason: str | None = None

    @property
    def positive_prompt(self) -> str:
        return (
            "### Reference Sheet Description\n"
            f"{self.reference_description}\n\n"
            "### Target Description\n"
            f"{self.shot_invariants}"
        )


def build_ingredients_vision_prompt(
    *,
    llm: VisionLLMPort | None,
    references: list[ReferenceImage],
    reference_metadata: list[dict[str, str]],
    target_context: dict[str, Any],
    fallback_reference_description: str,
    fallback_shot_invariants: str,
    scene_sheet_description: str = "",
    dspy_runtime: Any | None = None,
    image_factory=None,
) -> IngredientsPromptResult:
    unavailable_fallback = IngredientsPromptResult(
        fallback_reference_description, fallback_shot_invariants, "vision unavailable"
    )
    invalid_fallback = IngredientsPromptResult(
        fallback_reference_description, fallback_shot_invariants, "invalid response"
    )
    probe_failed_fallback = IngredientsPromptResult(
        fallback_reference_description, fallback_shot_invariants, "vision probe failed"
    )
    if llm is None:
        return unavailable_fallback

    capability_check = getattr(llm, "model_supports_vision", None)
    if callable(capability_check):
        try:
            supports_vision = capability_check()
        except Exception as exc:
            logger.warning(
                "Vision capability probe failed (%s); using text-only fallback",
                type(exc).__name__,
            )
            return probe_failed_fallback
        if not supports_vision:
            logger.warning(
                "Vision capability probe reports no vision support; using text-only fallback"
            )
            return unavailable_fallback

    try:
        result = IngredientsPromptModules(llm, dspy_runtime=dspy_runtime, image_factory=image_factory).vision(
            {
                "references": reference_metadata,
                "target_context": target_context,
                "scene_sheet_description": scene_sheet_description,
            },
            [reference.path for reference in references],
        )
    except (ConnectionError, TimeoutError, OSError, RuntimeError, FeverSlopLMLError):
        return unavailable_fallback
    except (TypeError, ValueError, KeyError):
        return invalid_fallback
    except Exception:
        return unavailable_fallback

    expected_pairs = {(reference.id, reference.type) for reference in references}
    descriptions: dict[tuple[str, str], str] = {}
    positions: dict[tuple[str, str], str] = {}
    for item in result.references:
        pair = (item.id, item.type)
        if pair in descriptions or not item.t2i_description.strip():
            return invalid_fallback
        descriptions[pair] = item.t2i_description.strip()
        if item.position.strip():
            positions[pair] = item.position.strip()
    if set(descriptions) != expected_pairs or len(descriptions) != len(references):
        return invalid_fallback

    reference_lines = []
    for reference in references:
        desc = descriptions[(reference.id, reference.type)]
        pos = positions.get((reference.id, reference.type))
        label = _reference_label(reference.type)
        if pos:
            reference_lines.append(
                f"{label} `{reference.id}` ({pos}): {desc}"
            )
        else:
            reference_lines.append(
                f"{label} `{reference.id}`: {desc}"
            )
    reference_lines.append(
        "The source images provide appearance only; do not reproduce their framing, composition, "
        "borders, panels, or layout."
    )
    return IngredientsPromptResult("\n".join(reference_lines), result.shot_invariants.strip())


def _reference_label(reference_type: str) -> str:
    return "Character" if reference_type == "actor" else "Setting"
