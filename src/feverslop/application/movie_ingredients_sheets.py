from __future__ import annotations

import json
import logging
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from feverslop.application.reference_bible import (
    INGREDIENTS_SHEET_LAYOUT_VERSION,
    build_ingredients_target_binding,
    build_runtime_consistency_contract,
    compose_cached_ingredients_sheet,
    generate_scene_sheet_description,
    generate_scene_sheet_anchors,
    ingredients_signature_references,
    ingredients_signature_sources,
    ingredients_sheet_size,
    snapshot_ingredients_sources,
    visual_consistency_sources,
)
from feverslop.domain.prepared_workflow import sha256_file
from feverslop.domain.visual_consistency_runtime import (
    bind_continuity_anchors,
    reference_look_id,
    resolve_reference_look,
)
from feverslop.application.ingredients_render_plan import build_ingredients_static_prompt
from feverslop.application.movie_msr_enrichment import _movie_video_prompt
from feverslop.ports.llm import VisionLLMPort
from feverslop.application.ingredients_vision_prompt import build_ingredients_vision_prompt
from feverslop.domain.vision_references import ReferenceImage


logger = logging.getLogger(__name__)


def enrich_movie_render_plan_with_ingredients_sheets(
    *,
    project_dir: Path,
    sheet_scale: float = 2.0,
    llm: VisionLLMPort | None = None,
    on_analysis_status: Callable[[str, list[dict[str, str]]], None] | None = None,
    workflow_profile: str = "ingredients-default",
) -> Path:
    """Compose per-shot Ingredients scene reference sheets and write
    movie/render_plan_ingredients.json.

    Reads render_plan.json + references/manifest.json, composes letterboxed
    scene sheets, generates structured descriptions, and persists the result.
    Fully independent of the MSR pipeline.

    Parameters
    ----------
    project_dir: Project root.
    sheet_scale: Minimum multiplier over the project resolution. The resulting
                 canvas is expanded to the Ingredients model's 12:7 aspect.
    """
    project_dir = Path(project_dir)
    movie_dir = project_dir / "movie"
    render_plan_path = movie_dir / "render_plan.json"
    reference_manifest_path = movie_dir / "references" / "manifest.json"
    continuity_plan_path = movie_dir / "continuity_plan.json"
    shot_cards_path = movie_dir / "shot_cards.json"
    bible_path = movie_dir / "bible.json"
    render_plan = _read_json(render_plan_path)
    manifest = _read_json(reference_manifest_path)
    try:
        bible = _read_json(bible_path)
    except (FileNotFoundError, IsADirectoryError):
        bible = {}
    try:
        continuity_plan = _read_json(continuity_plan_path)
    except (FileNotFoundError, IsADirectoryError):
        continuity_plan = {}
    try:
        shot_cards = _read_json(shot_cards_path)
    except (FileNotFoundError, IsADirectoryError):
        shot_cards = {}

    base_w, base_h = _read_json(render_plan_path).get("resolution", {}).get("width", 1280), \
        _read_json(render_plan_path).get("resolution", {}).get("height", 704)
    sheet_size = ingredients_sheet_size(base_w, base_h, sheet_scale)

    enriched = deepcopy(render_plan)
    enriched["movie_bible_path"] = "movie/bible.json"
    if continuity_plan:
        enriched["movie_continuity_plan_path"] = "movie/continuity_plan.json"
    enriched["reference_manifest_path"] = "movie/references/manifest.json"
    if shot_cards:
        enriched["movie_shot_cards_path"] = "movie/shot_cards.json"
    enriched["ingredients_enriched"] = True

    builder = IngredientsSceneSheetBuilder(
        project_dir=project_dir,
        manifest=manifest,
        size=sheet_size,
    )
    fps = int(render_plan.get("fps") or (bible.get("runtime_constraints") or {}).get("fps") or 24)
    enriched["shots"] = [
        _enrich_shot(
            {
                **shot,
                "scene": int(shot.get("scene") or index),
            },
            builder=builder,
            manifest=manifest,
            bible=bible,
            fps=fps,
            llm=llm,
            on_analysis_status=on_analysis_status,
            workflow_profile=workflow_profile,
        )
        for index, shot in enumerate(render_plan.get("shots") or [], start=1)
    ]

    output_path = movie_dir / "render_plan_ingredients.json"
    output_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def _enrich_shot(
    shot: dict,
    *,
    builder: "IngredientsSceneSheetBuilder",
    manifest: dict,
    bible: dict,
    fps: int,
    llm: VisionLLMPort | None,
    on_analysis_status: Callable[[str, list[dict[str, str]]], None] | None,
    workflow_profile: str,
) -> dict:
    enriched = deepcopy(shot)
    sheet_result = builder.build(shot)
    enriched["ingredients_scene_sheet"] = sheet_result["sheet_path"]
    enriched["ingredients_scene_sheet_description"] = sheet_result.get("scene_reference_sheet_description", "")
    anchors = list(sheet_result.get("scene_reference_sheet_anchors") or [])
    enriched["ingredients_scene_sheet_anchors"] = anchors
    if images := list(sheet_result.get("images") or []):
        contract = build_runtime_consistency_contract(
            shot,
            images=images,
            project_base=builder.project_dir,
            mode="ingredients",
            workflow_profile=workflow_profile,
        )
        enriched["visual_consistency"] = contract.to_dict()
        enriched["ingredients_sheet_signature"] = sheet_result["signature"]
        enriched["ingredients_sheet_layout_version"] = sheet_result["layout_version"]
        enriched["ingredients_sheet_size"] = sheet_result["size"]
        enriched["ingredients_signature_references"] = sheet_result[
            "signature_references"
        ]
        enriched["ingredients_signature_sources"] = sheet_result[
            "signature_sources"
        ]
        enriched["ingredients_sheet_sha256"] = sheet_result["sheet_sha256"]
        enriched["visual_consistency_sources"] = sheet_result[
            "visual_consistency_sources"
        ]
    fallback_invariants = _fallback_movie_shot_invariants(shot, anchors=anchors)
    references = [
        ReferenceImage(id=str(img["id"]), type=img["type"], path=builder.project_dir / img["path"])
        for img in images
    ]
    shot_id = str(shot.get("shot_id") or shot.get("scene") or "")
    status_references = [{"id": ref.id, "type": ref.type} for ref in references]
    if references:
        logger.info("Ingredients image analysis attempt: shot=%s reference_count=%s references=%s", shot_id, len(references), status_references)
    if llm is not None and references and on_analysis_status is not None:
        on_analysis_status(shot_id, status_references)
    result = build_ingredients_vision_prompt(
        llm=llm if references else None,
        references=references,
        reference_metadata=[
            {key: str(img.get(key) or "") for key in ("id", "type", "name", "visual_description", "image_prompt")}
            for img in images
        ],
        target_context=_movie_target_context(shot, bible),
        fallback_reference_description=enriched["ingredients_scene_sheet_description"],
        fallback_shot_invariants=fallback_invariants,
    )
    global_prompt = bind_continuity_anchors(
        result.positive_prompt,
        enriched.get("visual_consistency"),
    )
    enriched["ingredients_global_prompt"] = global_prompt
    ltx = dict(enriched.get("ltx") or {})
    relay = list(ltx.get("msr_prompt_relay") or ltx.get("prompt_relay") or [])
    if not relay:
        duration = float(shot.get("duration_seconds") or shot.get("duration") or 1.0)
        frame_count = int(shot.get("frame_count") or max(1, round(duration * fps)))
        relay = [{
            "frame_start": 0,
            "frame_end": frame_count,
            "state": "dialogue" if str(shot.get("dialogue") or "").strip() else "motion",
            "prompt": _movie_video_prompt(shot, bible=bible, manifest=manifest),
        }]
    enriched["ingredients"] = {
        "sheet_path": enriched["ingredients_scene_sheet"],
        "anchors": anchors,
        "global_prompt": global_prompt,
        **(
            {
                "signature": sheet_result["signature"],
                "layout_version": sheet_result["layout_version"],
                "size": sheet_result["size"],
                "signature_references": sheet_result["signature_references"],
                "signature_sources": sheet_result["signature_sources"],
                "sheet_sha256": sheet_result["sheet_sha256"],
            }
            if references
            else {}
        ),
    }
    if not references:
        logger.warning("Ingredients image analysis fallback: shot=%s reason=no images", shot_id)
    elif result.fallback_reason:
        logger.warning(
            "Ingredients image analysis fallback: shot=%s reason=%s",
            shot_id,
            result.fallback_reason,
        )
    enriched["ltx"] = {
        **ltx,
        "base_prompt": global_prompt,
        "static_prompt": build_ingredients_static_prompt(global_prompt, relay),
        "prompt_relay": relay,
        "native_audio": True,
    }
    return enriched


class IngredientsSceneSheetBuilder:
    def __init__(
        self,
        *,
        project_dir: str | Path,
        manifest: dict,
        size: tuple[int, int] = (1280, 704),
    ):
        self.project_dir = Path(project_dir)
        self.manifest = manifest
        self.size = size

    def build(self, shot: dict) -> dict:
        actor_ids = shot.get("reference_ids", {}).get("actors") or shot.get("actor_ids") or []
        location_id = shot.get("reference_ids", {}).get("location") or shot.get("location_id") or ""

        images = []
        for actor_id in actor_ids:
            actor_item = _item_for_id(self.manifest.get("actors") or [], actor_id)
            if actor_item:
                actor_item = resolve_reference_look(
                    actor_item,
                    reference_look_id(
                        shot,
                        kind="actor",
                        semantic_id=str(actor_id),
                    ),
                )
                logical = _pick_existing_path(actor_item.get("sheet_path"), self.project_dir)
                if logical:
                    images.append({
                        "path": logical,
                        "contract_path": _pick_existing_path(
                            (
                                logical
                                if actor_item.get("look_id") != "default"
                                else (
                                    actor_item.get("msr_sheet_path")
                                    or actor_item.get("msr_input_path")
                                    or logical
                                )
                            ),
                            self.project_dir,
                        ) or logical,
                        "type": "actor",
                        "id": actor_id,
                        "look_id": str(actor_item.get("look_id") or "default"),
                        "visual_description": str(actor_item.get("visual_description") or "").strip(),
                        "name": str(actor_item.get("name") or "").strip(),
                        "image_prompt": str(actor_item.get("image_prompt") or "").strip(),
                    })

        if location_id:
            location_item = _item_for_id(self.manifest.get("locations") or [], location_id)
            if location_item:
                location_item = resolve_reference_look(
                    location_item,
                    reference_look_id(
                        shot,
                        kind="location",
                        semantic_id=str(location_id),
                    ),
                )
                logical = _pick_existing_path(location_item.get("sheet_path"), self.project_dir)
                if logical:
                    images.append({
                        "path": logical,
                        "contract_path": _pick_existing_path(
                            (
                                logical
                                if location_item.get("look_id") != "default"
                                else (
                                    location_item.get("msr_sheet_path")
                                    or location_item.get("msr_background_path")
                                    or logical
                                )
                            ),
                            self.project_dir,
                        ) or logical,
                        "type": "location",
                        "id": location_id,
                        "look_id": str(location_item.get("look_id") or "default"),
                        "visual_description": str(location_item.get("visual_description") or "").strip(),
                        "name": str(location_item.get("name") or "").strip(),
                        "image_prompt": str(location_item.get("image_prompt") or "").strip(),
                    })

        if not images:
            return {
                "sheet_path": "",
                "image_count": 0,
                "images": [],
                "scene_reference_sheet_description": "",
                "scene_reference_sheet_anchors": [],
            }

        image_paths = [self.project_dir / img["path"] for img in images]
        source_snapshots = snapshot_ingredients_sources(
            images,
            project_base=self.project_dir,
        )
        signature_references = ingredients_signature_references(
            images,
            project_base=self.project_dir,
            snapshots=source_snapshots,
        )
        signature_sources = ingredients_signature_sources(
            images,
            project_base=self.project_dir,
            snapshots=source_snapshots,
        )
        contract_sources = visual_consistency_sources(
            images,
            project_base=self.project_dir,
        )
        output_path, signature = compose_cached_ingredients_sheet(
            image_paths,
            cache_dir=(
                self.project_dir
                / "movie"
                / "references"
                / "ingredients_sheets"
                / "by_signature"
            ),
            references=signature_references,
            size=self.size,
            snapshots=source_snapshots,
        )
        num_cols = math.ceil(math.sqrt(len(images)))

        relative_sheet = output_path.relative_to(self.project_dir).as_posix()
        description = generate_scene_sheet_description(images, num_cols, self.size)
        anchors = generate_scene_sheet_anchors(images, num_cols)
        return {
            "sheet_path": relative_sheet,
            "image_count": len(images),
            "images": images,
            "signature": signature,
            "layout_version": INGREDIENTS_SHEET_LAYOUT_VERSION,
            "size": list(self.size),
            "signature_references": signature_references,
            "signature_sources": signature_sources,
            "sheet_sha256": sha256_file(output_path),
            "visual_consistency_sources": contract_sources,
            "scene_reference_sheet_description": description,
            "scene_reference_sheet_anchors": anchors,
        }


def _movie_target_context(shot: dict, bible: dict) -> dict[str, Any]:
    return {
        "description": shot.get("description") or "",
        "action": shot.get("action") or "",
        "camera": shot.get("camera") or shot.get("camera_motion") or "",
        "acting": shot.get("acting") or "",
        "dialogue": shot.get("dialogue") or "",
        "continuity_notes": shot.get("continuity_notes") or "",
        "duration_seconds": shot.get("duration_seconds") or shot.get("duration") or "",
        "dialogue_policy": (bible.get("runtime_constraints") or {}).get("dialogue_language") or "",
    }


def _fallback_movie_shot_invariants(shot: dict, *, anchors: list[dict]) -> str:
    parts = [
        build_ingredients_target_binding(anchors).strip(),
        "Maintain one continuous full-frame shot with stable identities, wardrobe, spatial staging, environment, and lighting.",
    ]
    camera = shot.get("camera") or shot.get("camera_motion") or ""
    action = shot.get("action") or ""
    acting = shot.get("acting") or shot.get("expression") or ""
    if action:
        parts.append(f"Action: {str(action).strip()}")
    if camera:
        parts.append(f"Camera policy: {str(camera).strip()}")
    if acting:
        parts.append(f"Acting: {str(acting).strip()}")
    return " ".join(part for part in parts if part).strip()


def _pick_existing_path(value: Any | None, project_dir: Path) -> str:
    if not value:
        return ""
    candidate = str(value).strip()
    if not candidate:
        return ""
    full = project_dir / candidate
    if full.exists():
        return candidate
    return ""


def _item_for_id(items: list[dict], item_id: str) -> dict | None:
    for item in items:
        if isinstance(item, dict) and str(item.get("id")) == str(item_id):
            return item
    return None


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Movie pipeline artifact not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Movie pipeline artifact must be a JSON object: {path}")
    return data
