from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable
import json
import logging
import math

from feverslop.application.reference_bible import (
    build_ingredients_target_binding,
    compose_scene_reference_sheet,
    generate_scene_sheet_description,
    generate_scene_sheet_anchors,
    ingredients_sheet_size,
)
from feverslop.ports.llm import VisionLLMPort
from feverslop.application.ingredients_vision_prompt import build_ingredients_vision_prompt
from feverslop.domain.vision_references import ReferenceImage


logger = logging.getLogger(__name__)


def enrich_render_plan_with_ingredients_sheets(
    render_plan_path: str | Path,
    references_dir: str | Path,
    output_path: str | Path,
    *,
    video_settings: Any = None,
    sheet_scale: float = 2.0,
    llm: VisionLLMPort | None = None,
    on_analysis_status: Callable[[int, list[dict[str, str]]], None] | None = None,
    on_scene_complete: Callable[[int, int, int], None] | None = None,
) -> Path:
    """Compose per-scene Ingredients reference sheets for song-based render plans.

    Reads render plan (list of scenes with references), composes letterboxed
    scene sheets from actor + location reference images, and generates
    structured descriptions. Writes the enriched plan to output_path.

    Mirrors enrich_render_plan_with_reference_sheets for MSR but produces
    ingredients-specific fields: ingredients_scene_sheet, ingredients_scene_sheet_description,
    ingredients_target_prompt, and ltx.ingredients_* fields.
    """
    render_plan_path = Path(render_plan_path)
    references_dir = Path(references_dir)
    output_path = Path(output_path)
    render_plan = json.loads(render_plan_path.read_text(encoding="utf-8-sig"))

    actor_manifests = _load_manifests_by_id(references_dir / "actors")
    location_manifests = _load_manifests_by_id(references_dir / "locations")

    first_scene = render_plan[0] if render_plan else {}
    if video_settings:
        base_w = video_settings.width
        base_h = video_settings.height
    else:
        base_w = int(first_scene.get("width", 1280))
        base_h = int(first_scene.get("height", 704))
    sheet_size = ingredients_sheet_size(base_w, base_h, sheet_scale)

    project_base = _infer_reference_project_base(references_dir)

    total = len(render_plan)
    enriched_scenes = []
    for index, scene in enumerate(render_plan, start=1):
        enriched = _enrich_scene(
            scene,
            project_base=project_base,
            actor_manifests=actor_manifests,
            location_manifests=location_manifests,
            sheet_size=sheet_size,
            references_dir=references_dir,
            llm=llm,
            on_analysis_status=on_analysis_status,
        )
        enriched_scenes.append(enriched)
        if on_scene_complete is not None:
            on_scene_complete(int(scene.get("scene", index)), index, total)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(enriched_scenes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def _enrich_scene(
    scene: dict,
    *,
    project_base: Path,
    actor_manifests: dict[str, dict],
    location_manifests: dict[str, dict],
    sheet_size: tuple[int, int],
    references_dir: Path,
    llm: VisionLLMPort | None,
    on_analysis_status: Callable[[int, list[dict[str, str]]], None] | None,
) -> dict:
    enriched = deepcopy(scene)
    references = enriched.get("references") or {}

    actor_ids = list(references.get("actor_ids") or [])
    images = []
    for actor_id in actor_ids:
        manifest = actor_manifests.get(str(actor_id))
        if manifest:
            sheet_path = str(manifest.get("sheet_path") or "").strip()
            if sheet_path and (project_base / sheet_path).exists():
                desc = str(manifest.get("visual_description") or "").strip()
                images.append({
                    "path": sheet_path,
                    "type": "actor",
                    "id": str(actor_id),
                    "visual_description": desc,
                    "name": str(manifest.get("name") or "").strip(),
                    "image_prompt": str(manifest.get("image_prompt") or "").strip(),
                })

    location_id = str(references.get("location_id") or "").strip()
    if location_id:
        manifest = location_manifests.get(location_id)
        if manifest:
            sheet_path = str(manifest.get("sheet_path") or "").strip()
            if sheet_path and (project_base / sheet_path).exists():
                desc = str(manifest.get("visual_description") or "").strip()
                images.append({
                    "path": sheet_path,
                    "type": "location",
                    "id": location_id,
                    "visual_description": desc,
                    "name": str(manifest.get("name") or "").strip(),
                    "image_prompt": str(manifest.get("image_prompt") or "").strip(),
                })

    scene_number = int(scene.get("scene", 0))
    if images:
        image_paths = [project_base / img["path"] for img in images]
        output_dir = references_dir / "ingredients_sheets"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"scene_{scene_number:04}_ingredients.png"

        num_cols = math.ceil(math.sqrt(len(images)))
        compose_scene_reference_sheet(image_paths, output_path, size=sheet_size)

        try:
            relative_sheet = output_path.relative_to(project_base).as_posix()
        except ValueError:
            relative_sheet = output_path.as_posix()

        description = generate_scene_sheet_description(images, num_cols, sheet_size)
        anchors = generate_scene_sheet_anchors(images, num_cols)
        enriched["ingredients_scene_sheet"] = relative_sheet
        enriched["ingredients_scene_sheet_description"] = description
        enriched["ingredients_scene_sheet_anchors"] = anchors
    else:
        enriched["ingredients_scene_sheet"] = ""
        enriched["ingredients_scene_sheet_description"] = ""
        enriched["ingredients_scene_sheet_anchors"] = []

    ltx = scene.get("ltx") or {}
    target_prompt = str(ltx.get("i2v_prompt_from_t2i") or "").strip()
    binding = build_ingredients_target_binding(enriched.get("ingredients_scene_sheet_anchors") or [])
    fallback_target = binding + target_prompt if target_prompt else ""
    if images:
        references = [ReferenceImage(id=img["id"], type=img["type"], path=project_base / img["path"]) for img in images]
        status_references = [{"id": ref.id, "type": ref.type} for ref in references]
        logger.info("Ingredients image analysis attempt: scene=%s reference_count=%s references=%s", scene_number, len(references), status_references)
        if llm is not None and on_analysis_status is not None:
            on_analysis_status(scene_number, status_references)
        result = build_ingredients_vision_prompt(
            llm=llm,
            references=references,
            reference_metadata=[{key: str(img.get(key) or "") for key in ("id", "type", "name", "visual_description", "image_prompt")} for img in images],
            target_context=_song_target_context(scene),
            fallback_reference_description=enriched.get("ingredients_scene_sheet_description", ""),
            fallback_target_prompt=fallback_target,
        )
        enriched["ingredients_scene_sheet_description"] = (
            "### Reference Sheet Description\n" + result.reference_description
        )
        enriched["ingredients_target_prompt"] = (
            "### Target Description\n" + result.target_description if result.target_description else ""
        )
        if result.fallback_reason:
            logger.warning(
                "Ingredients image analysis fallback: scene=%s reason=%s",
                scene_number,
                result.fallback_reason,
            )
    else:
        logger.warning("Ingredients image analysis fallback: scene=%s reason=no images", scene_number)
        enriched["ingredients_target_prompt"] = "### Target Description\n" + fallback_target if fallback_target else ""

    enriched_ltx = dict(enriched.get("ltx") or {})
    enriched_ltx["ingredients_scene_sheet_description"] = enriched.get("ingredients_scene_sheet_description", "")
    enriched_ltx["ingredients_target_prompt"] = enriched.get("ingredients_target_prompt", "")
    enriched_ltx["native_audio"] = True
    enriched["ltx"] = enriched_ltx

    return enriched


def _song_target_context(scene: dict) -> dict[str, Any]:
    ltx = scene.get("ltx") or {}
    return {
        "source_video_prompt": ltx.get("i2v_prompt_from_t2i") or "",
        "concept": scene.get("concept") or scene.get("description") or "",
        "metadata": scene.get("metadata") or {},
        "camera_motion": scene.get("camera_motion") or scene.get("camera") or "",
        "character_motion": scene.get("character_motion") or scene.get("action") or "",
        "lyrics": scene.get("lyrics") or scene.get("dialogue") or "",
        "dialogue_policy": scene.get("dialogue_policy") or "",
        "duration_seconds": scene.get("duration_seconds") or scene.get("duration") or "",
    }


def _load_manifests_by_id(root: Path) -> dict[str, dict]:
    manifests: dict[str, dict] = {}
    if not root.exists():
        return manifests
    for manifest_path in root.glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        manifests[str(manifest["id"])] = manifest
    return manifests


def _infer_reference_project_base(references_dir: Path) -> Path:
    if references_dir.name == "references" and references_dir.parent.name == "output":
        return references_dir.parent.parent
    return references_dir
