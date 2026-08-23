from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from feverslop.domain.visual_consistency_runtime import (
    bind_continuity_anchors,
    scrub_prior_context,
)
from feverslop.errors import FeverSlopValidationError
from feverslop.prompting.subject_directive_planning import (
    subject_directives_from_scene,
)
from feverslop.prompting.subject_directive_projections import project_subject_directives


_CORE_SCENE_FIELDS = (
    "scene",
    "abs_start_seconds",
    "abs_end_seconds",
    "duration_seconds",
    "fps",
    "width",
    "height",
    "frame_count",
    "cut",
)
_CORE_METADATA_FIELDS = ("segment_id", "type", "silent_mode", "lyrics")


def project_ingredients_runtime_scene(scene: Mapping[str, Any]) -> dict[str, Any]:
    ltx = scene.get("ltx") or {}
    relay = deepcopy(ltx.get("msr_prompt_relay") or ltx.get("prompt_relay") or [])
    visual_consistency = deepcopy(scene.get("visual_consistency"))
    global_prompt = bind_continuity_anchors(
        _ingredients_global_prompt(scene),
        visual_consistency,
    )
    directive_plan = subject_directives_from_scene(scene)
    if directive_plan is not None:
        global_prompt = f"{global_prompt}\n\n{project_subject_directives(directive_plan, backend='ltx-ingredients').prompt}"
    scene_number = scene.get("scene", "?")
    if not global_prompt:
        raise FeverSlopValidationError(f"Scene {scene_number} is missing the Ingredients global prompt")
    if not relay:
        raise FeverSlopValidationError(f"Scene {scene_number} is missing the Ingredients prompt relay")

    projected = _present_fields(scene, _CORE_SCENE_FIELDS)
    projected["metadata"] = _present_fields(scene.get("metadata") or {}, _CORE_METADATA_FIELDS)
    references = scene.get("references") or {}
    projected["references"] = {
        "actor_ids": deepcopy(list(references.get("actor_ids") or [])),
        "location_id": str(references.get("location_id") or ""),
    }
    projected["ingredients"] = {
        "sheet_path": str(scene.get("ingredients_scene_sheet") or "").strip(),
        "anchors": deepcopy(list(scene.get("ingredients_scene_sheet_anchors") or [])),
        "global_prompt": global_prompt,
    }
    if scene.get("ingredients_sheet_signature"):
        projected["ingredients"].update({
            "signature": str(scene["ingredients_sheet_signature"]),
            "layout_version": str(scene.get("ingredients_sheet_layout_version") or ""),
            "size": deepcopy(list(scene.get("ingredients_sheet_size") or [])),
            "signature_references": deepcopy(
                list(scene.get("ingredients_signature_references") or [])
            ),
            "signature_sources": deepcopy(
                list(scene.get("ingredients_signature_sources") or [])
            ),
            "sheet_sha256": str(scene.get("ingredients_sheet_sha256") or ""),
        })
    if visual_consistency is not None:
        projected["visual_consistency"] = visual_consistency
        projected["visual_consistency_sources"] = deepcopy(
            scene.get("visual_consistency_sources") or {}
        )
    projected["ltx"] = {
        "base_prompt": global_prompt,
        "static_prompt": build_ingredients_static_prompt(global_prompt, relay),
        "prompt_relay": relay,
        "native_audio": True,
    }
    return projected


def build_ingredients_static_prompt(global_prompt: str, relay: list[dict[str, Any]]) -> str:
    states = [str(segment.get("state") or "").strip().lower() for segment in relay]
    if states and all(state != "singing" for state in states):
        policy = "No vocal performance throughout; mouths remain closed for the entire shot."
    else:
        phases = []
        for index, segment in enumerate(relay):
            state = str(segment.get("state") or "motion").strip().lower()
            prefix = "At the start" if index == 0 else f"Then {state}"
            phases.append(f"{prefix}: {str(segment.get('prompt') or '').strip()}")
        policy = "Temporal sequence (best effort in a static workflow): " + " ".join(phases)
    return scrub_prior_context(f"{global_prompt.strip()}\n{policy}".strip())


def _ingredients_global_prompt(scene: Mapping[str, Any]) -> str:
    ingredients = scene.get("ingredients") or {}
    ltx = scene.get("ltx") or {}
    return str(
        ingredients.get("global_prompt")
        or scene.get("ingredients_global_prompt")
        or scene.get("ingredients_scene_sheet_description")
        or ltx.get("ingredients_global_prompt")
        or ltx.get("ingredients_scene_sheet_description")
        or ""
    ).strip()


def _present_fields(source: Mapping[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    return {name: deepcopy(source[name]) for name in names if name in source}
