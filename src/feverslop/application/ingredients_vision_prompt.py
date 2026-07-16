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
  "target_description": "250-400 word continuous-shot direction"
}

Treat the supplied images as ground truth. Text metadata is supplementary intent only.
Describe stable visible identity and environment details, but omit source pose, source camera,
borders, panels, labels, typography, and sheet layout. Do not reproduce the source framing,
composition, borders, panels, or layout.

Write target_description in 250-400 words. It must describe one single continuous full-frame shot
with a concrete opening, chronological progression, and final state. Repeat identity-critical
details and specify spatial staging, chronological action, facial acting, body motion,
clothing and hair response, camera start/motion/end, environment motion, and lighting behavior.
Do not include captions, titles, signs, logos, screens, UI/HUD, or written characters unless the
shot context explicitly requires them. Include every supplied reference exactly once with its
unchanged id and type.
"""


@dataclass(frozen=True)
class IngredientsPromptResult:
    reference_description: str
    target_description: str
    fallback_reason: str | None = None

    @property
    def positive_prompt(self) -> str:
        return (
            "### Reference Sheet Description\n"
            f"{self.reference_description}\n\n"
            "### Target Description\n"
            f"{self.target_description}"
        )


def build_ingredients_vision_prompt(
    *,
    llm: VisionLLMPort | None,
    references: list[ReferenceImage],
    reference_metadata: list[dict[str, str]],
    target_context: dict[str, Any],
    fallback_reference_description: str,
    fallback_target_prompt: str,
) -> IngredientsPromptResult:
    unavailable_fallback = IngredientsPromptResult(
        fallback_reference_description, fallback_target_prompt, "vision unavailable"
    )
    invalid_fallback = IngredientsPromptResult(
        fallback_reference_description, fallback_target_prompt, "invalid response"
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
        target = data.get("target_description")
        expected_pairs = {(reference.id, reference.type) for reference in references}
        if not isinstance(parsed_references, list) or len(parsed_references) != len(references):
            return invalid_fallback
        if not isinstance(target, str) or len(target.split()) < 180:
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
    bindings = " ".join(
        f"Use {_reference_label(reference.type)} `{reference.id}` as "
        f"{'a visible character.' if reference.type == 'actor' else 'the environment.'}"
        for reference in references
    )
    return IngredientsPromptResult("\n".join(reference_lines), f"{bindings}\n{target.strip()}")


def _reference_label(reference_type: str) -> str:
    return "Character" if reference_type == "actor" else "Setting"
