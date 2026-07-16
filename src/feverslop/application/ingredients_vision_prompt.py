from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from feverslop.domain.llm_parsing import extract_json_object
from feverslop.ports.llm import VisionLLMPort
from feverslop.domain.vision_references import ReferenceImage


_SYSTEM_PROMPT = """You create a vision-grounded Ingredients prompt for video generation.
Return only JSON with this exact shape:
{
  "references": [
    {"id": "reference id", "type": "actor or location", "description": "stable visible identity details"}
  ],
  "shot_invariants": "60-160 word non-temporal continuous-shot contract"
}

Treat the supplied images as ground truth. Text metadata is supplementary intent only.
Describe stable visible identity and environment details, but omit source pose, source camera,
borders, panels, labels, typography, and sheet layout. Do not reproduce the source framing,
composition, borders, panels, or layout.

Write shot_invariants in 60-160 words. It must describe one single continuous full-frame shot.
Specify stable spatial staging, camera framing and motion policy, identity-critical details,
clothing and hair behavior, environment motion, and lighting behavior. Keep it non-temporal.
Do not schedule an opening, progression, final state, dialogue timing, singing, lip-sync,
mouth state, or any other performance transition; those are supplied by a frame-level relay.
Do not include captions, titles, signs, logos, screens, UI/HUD, or written characters unless the
shot context explicitly requires them. Include every supplied reference exactly once with its
unchanged id and type.
"""


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
            "### Shot Invariants\n"
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
) -> IngredientsPromptResult:
    unavailable_fallback = IngredientsPromptResult(
        fallback_reference_description, fallback_shot_invariants, "vision unavailable"
    )
    invalid_fallback = IngredientsPromptResult(
        fallback_reference_description, fallback_shot_invariants, "invalid response"
    )
    if llm is None:
        return unavailable_fallback

    payload = json.dumps(
        {"references": reference_metadata, "target_context": target_context},
        ensure_ascii=True,
    )
    try:
        response = llm.complete_prompt_with_images(
            _SYSTEM_PROMPT,
            payload,
            [reference.path for reference in references],
        )
    except Exception:
        return unavailable_fallback

    try:
        data = extract_json_object(response)
        parsed_references = data.get("references")
        shot_invariants = data.get("shot_invariants")
        expected_pairs = {(reference.id, reference.type) for reference in references}
        if not isinstance(parsed_references, list) or len(parsed_references) != len(references):
            return invalid_fallback
        if not isinstance(shot_invariants, str) or not 60 <= len(shot_invariants.split()) <= 160:
            return invalid_fallback

        descriptions: dict[tuple[str, str], str] = {}
        for item in parsed_references:
            if not isinstance(item, dict):
                return invalid_fallback
            reference_id = item.get("id")
            reference_type = item.get("type")
            description = item.get("description")
            if not all(isinstance(value, str) and value.strip() for value in (reference_id, reference_type, description)):
                return invalid_fallback
            descriptions[(reference_id, reference_type)] = description.strip()
        if set(descriptions) != expected_pairs or len(descriptions) != len(references):
            return invalid_fallback
    except Exception:
        return invalid_fallback

    reference_lines = [
        f"{_reference_label(reference.type)} `{reference.id}`: {descriptions[(reference.id, reference.type)]}"
        for reference in references
    ]
    reference_lines.append(
        "The source images provide appearance only; do not reproduce their framing, composition, "
        "borders, panels, or layout."
    )
    return IngredientsPromptResult("\n".join(reference_lines), shot_invariants.strip())


def _reference_label(reference_type: str) -> str:
    return "Character" if reference_type == "actor" else "Setting"
