from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

from feverslop.application.effective_render_plan import project_effective_plan
from feverslop.application.ingredients_render_plan import (
    project_ingredients_runtime_scene,
)
from feverslop.application.ingredients_vision_prompt import (
    build_ingredients_vision_prompt,
)
from feverslop.application.reference_bible import (
    INGREDIENTS_SHEET_LAYOUT_VERSION,
    build_ingredients_target_binding,
    build_runtime_consistency_contract,
    compose_cached_ingredients_sheet,
    generate_scene_sheet_anchors,
    generate_scene_sheet_description,
    ingredients_sheet_size,
    ingredients_signature_references,
    ingredients_signature_sources,
    snapshot_ingredients_sources,
    visual_consistency_sources,
)
from feverslop.domain.prepared_workflow import sha256_file
from feverslop.domain.vision_references import ReferenceImage
from feverslop.domain.visual_consistency_runtime import (
    bind_continuity_anchors,
    reference_look_id,
    resolve_reference_look,
)
from feverslop.ports.llm import VisionLLMPort
from feverslop.scene_artifacts import SceneArtifactLayout

logger = logging.getLogger(__name__)


def enrich_render_plan_with_ingredients_sheets(
    render_plan_path: str | Path,
    references_dir: str | Path,
    output_path: str | Path,
    *,
    canonical_plan_path: str | Path | None = None,
    video_settings: Any = None,
    sheet_scale: float = 2.0,
    llm: VisionLLMPort | None = None,
    on_analysis_status: Callable[[int, list[dict[str, str]]], None] | None = None,
    on_scene_complete: Callable[[int, int, int], None] | None = None,
    workflow_profile: str = "ingredients-default",
) -> Path:
    """Compose per-scene Ingredients reference sheets for song-based render plans.

    Reads render plan (list of scenes with references), composes letterboxed
    scene sheets from actor + location reference images, and generates
    structured descriptions. Writes a compact Ingredients runtime plan to output_path.

    Mirrors enrich_render_plan_with_reference_sheets for MSR but produces
    ingredients-specific sheet/global prompt data and a canonical temporal relay.
    """
    render_plan_path = Path(render_plan_path)
    references_dir = Path(references_dir)
    output_path = Path(output_path)
    render_plan = json.loads(render_plan_path.read_text(encoding="utf-8-sig"))
    if canonical_plan_path is not None:
        canonical_plan = json.loads(
            Path(canonical_plan_path).read_text(encoding="utf-8-sig"),
        )
        render_plan = project_effective_plan(render_plan, canonical_plan)

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
            workflow_profile=workflow_profile,
        )
        enriched_scenes.append(project_ingredients_runtime_scene(enriched))
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
    workflow_profile: str,
) -> dict:
    enriched = deepcopy(scene)
    references = enriched.get("references") or {}

    actor_ids = list(references.get("actor_ids") or [])
    images = []
    for actor_id in actor_ids:
        manifest = actor_manifests.get(str(actor_id))
        if manifest:
            manifest = resolve_reference_look(
                manifest,
                reference_look_id(
                    scene,
                    kind="actor",
                    semantic_id=str(actor_id),
                ),
            )
            sheet_path = str(manifest.get("sheet_path") or "").strip()
            if sheet_path and (project_base / sheet_path).exists():
                desc = str(manifest.get("visual_description") or "").strip()
                images.append({
                    "path": sheet_path,
                    "contract_path": str(
                        sheet_path
                        if manifest.get("look_id") != "default"
                        else (
                            manifest.get("msr_sheet_path")
                            or manifest.get("msr_input_path")
                            or sheet_path
                        ),
                    ).strip(),
                    "type": "actor",
                    "id": str(actor_id),
                    "look_id": str(manifest.get("look_id") or "default"),
                    "visual_description": desc,
                    "name": str(manifest.get("name") or "").strip(),
                    "image_prompt": str(manifest.get("image_prompt") or "").strip(),
                })

    location_id = str(references.get("location_id") or "").strip()
    if location_id:
        manifest = location_manifests.get(location_id)
        if manifest:
            manifest = resolve_reference_look(
                manifest,
                reference_look_id(
                    scene,
                    kind="location",
                    semantic_id=location_id,
                ),
            )
            sheet_path = str(manifest.get("sheet_path") or "").strip()
            if sheet_path and (project_base / sheet_path).exists():
                desc = str(manifest.get("visual_description") or "").strip()
                images.append({
                    "path": sheet_path,
                    "contract_path": str(
                        sheet_path
                        if manifest.get("look_id") != "default"
                        else (
                            manifest.get("msr_sheet_path")
                            or manifest.get("msr_background_path")
                            or sheet_path
                        ),
                    ).strip(),
                    "type": "location",
                    "id": location_id,
                    "look_id": str(manifest.get("look_id") or "default"),
                    "visual_description": desc,
                    "name": str(manifest.get("name") or "").strip(),
                    "image_prompt": str(manifest.get("image_prompt") or "").strip(),
                })

    scene_number = int(scene.get("scene", 0))
    if images:
        image_paths = [project_base / img["path"] for img in images]
        source_snapshots = snapshot_ingredients_sources(
            images,
            project_base=project_base,
        )
        signature_references = ingredients_signature_references(
            images,
            project_base=project_base,
            snapshots=source_snapshots,
        )
        signature_sources = ingredients_signature_sources(
            images,
            project_base=project_base,
            snapshots=source_snapshots,
        )
        cache_dir = SceneArtifactLayout(project_base).ingredients_sheet_cache_dir
        output_path, signature = compose_cached_ingredients_sheet(
            image_paths,
            cache_dir=cache_dir,
            references=signature_references,
            size=sheet_size,
            snapshots=source_snapshots,
        )

        num_cols = math.ceil(math.sqrt(len(images)))

        try:
            relative_sheet = output_path.relative_to(project_base).as_posix()
        except ValueError:
            relative_sheet = output_path.as_posix()

        description = generate_scene_sheet_description(images, num_cols, sheet_size)
        anchors = generate_scene_sheet_anchors(images, num_cols)
        enriched["ingredients_scene_sheet"] = relative_sheet
        enriched["ingredients_scene_sheet_description"] = description
        enriched["ingredients_scene_sheet_anchors"] = anchors
        enriched["ingredients_sheet_signature"] = signature
        enriched["ingredients_sheet_layout_version"] = INGREDIENTS_SHEET_LAYOUT_VERSION
        enriched["ingredients_sheet_size"] = list(sheet_size)
        enriched["ingredients_signature_references"] = signature_references
        enriched["ingredients_signature_sources"] = signature_sources
        enriched["ingredients_sheet_sha256"] = sha256_file(output_path)
        enriched["visual_consistency"] = build_runtime_consistency_contract(
            scene,
            images=images,
            project_base=project_base,
            mode="ingredients",
            workflow_profile=workflow_profile,
        ).to_dict()
        enriched["visual_consistency_sources"] = visual_consistency_sources(
            images,
            project_base=project_base,
        )
    else:
        enriched["ingredients_scene_sheet"] = ""
        enriched["ingredients_scene_sheet_description"] = ""
        enriched["ingredients_scene_sheet_anchors"] = []

    binding = build_ingredients_target_binding(enriched.get("ingredients_scene_sheet_anchors") or [])
    fallback_invariants = _fallback_shot_invariants(scene, binding=binding)
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
            fallback_shot_invariants=fallback_invariants,
            scene_sheet_description=enriched.get("ingredients_scene_sheet_description", ""),
        )
        enriched["ingredients_global_prompt"] = bind_continuity_anchors(
            result.positive_prompt,
            enriched.get("visual_consistency"),
        )
        if result.fallback_reason:
            logger.warning(
                "Ingredients image analysis fallback: scene=%s reason=%s",
                scene_number,
                result.fallback_reason,
            )
    else:
        logger.warning("Ingredients image analysis fallback: scene=%s reason=no images", scene_number)
        enriched["ingredients_global_prompt"] = ""

    return enriched


def _song_target_context(scene: dict) -> dict[str, Any]:
    ltx = scene.get("ltx") or {}
    metadata = scene.get("metadata") or {}
    return {
        "source_video_prompt": ltx.get("i2v_prompt_from_t2i") or "",
        "concept": scene.get("concept") or scene.get("description") or "",
        "metadata": metadata,
        "type": metadata.get("type") or scene.get("type") or "",
        "silent_mode": bool(metadata.get("silent_mode") or scene.get("silent_mode")),
        "camera_motion": metadata.get("camera_motion") or scene.get("camera_motion") or scene.get("camera") or "",
        "character_motion": metadata.get("character_motion") or scene.get("character_motion") or scene.get("action") or "",
        "lyrics": metadata.get("lyrics") or scene.get("lyrics") or scene.get("dialogue") or "",
        "dialogue_policy": scene.get("dialogue_policy") or "",
        "duration_seconds": scene.get("duration_seconds") or scene.get("duration") or "",
    }


def _fallback_shot_invariants(scene: dict, *, binding: str) -> str:
    metadata = scene.get("metadata") or {}
    camera = metadata.get("camera_motion") or scene.get("camera_motion") or scene.get("camera") or ""
    action = (
        metadata.get("character_motion")
        or scene.get("character_motion")
        or scene.get("action")
        or ""
    )
    lyrics = metadata.get("lyrics") or scene.get("lyrics") or ""
    parts = [
        binding.strip(),
        "Maintain one continuous full-frame shot with stable identities, wardrobe, spatial staging, environment, and lighting.",
    ]
    if action:
        parts.append(f"Action: {str(action).strip()}")
    if camera:
        parts.append(f"Camera policy: {str(camera).strip()}")
    if lyrics:
        parts.append(f"Lyric performance: {str(lyrics).strip()}")
    return " ".join(part for part in parts if part).strip()


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
