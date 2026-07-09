from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

from feverslop.application.reference_bible import (
    compose_scene_reference_sheet,
    generate_scene_sheet_description,
)


def enrich_movie_render_plan_with_ingredients_sheets(*, project_dir: Path) -> Path:
    """Compose per-shot Ingredients scene reference sheets and write
    movie/render_plan_ingredients.json.

    Reads render_plan.json + references/manifest.json, composes letterboxed
    scene sheets, generates structured descriptions, and persists the result.
    Fully independent of the MSR pipeline."""
    project_dir = Path(project_dir)
    movie_dir = project_dir / "movie"
    render_plan_path = movie_dir / "render_plan.json"
    reference_manifest_path = movie_dir / "references" / "manifest.json"
    continuity_plan_path = movie_dir / "continuity_plan.json"
    shot_cards_path = movie_dir / "shot_cards.json"
    render_plan = _read_json(render_plan_path)
    manifest = _read_json(reference_manifest_path)
    continuity_plan = _read_json(continuity_plan_path) if continuity_plan_path.exists() else {}
    shot_cards = _read_json(shot_cards_path) if shot_cards_path.exists() else {}

    enriched = deepcopy(render_plan)
    enriched["movie_bible_path"] = "movie/bible.json"
    if continuity_plan:
        enriched["movie_continuity_plan_path"] = "movie/continuity_plan.json"
    enriched["reference_manifest_path"] = "movie/references/manifest.json"
    if shot_cards:
        enriched["movie_shot_cards_path"] = "movie/shot_cards.json"
    enriched["ingredients_enriched"] = True

    builder = IngredientsSceneSheetBuilder(project_dir=project_dir, manifest=manifest)
    enriched["shots"] = [
        _enrich_shot(shot, builder=builder, manifest=manifest)
        for shot in render_plan.get("shots") or []
    ]

    output_path = movie_dir / "render_plan_ingredients.json"
    output_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def _enrich_shot(shot: dict, *, builder: "IngredientsSceneSheetBuilder", manifest: dict) -> dict:
    enriched = deepcopy(shot)
    sheet_result = builder.build(shot)
    enriched["ingredients_scene_sheet"] = sheet_result["sheet_path"]
    enriched["ingredients_scene_sheet_description"] = sheet_result.get("scene_reference_sheet_description", "")
    enriched["ltx"] = {
        **dict(enriched.get("ltx") or {}),
        "native_audio": True,
        "ingredients_scene_sheet_description": enriched.get("ingredients_scene_sheet_description", ""),
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
                logical = _pick_existing_path(actor_item.get("sheet_path"), self.project_dir)
                if logical:
                    images.append({
                        "path": logical,
                        "type": "actor",
                        "id": actor_id,
                        "visual_description": str(actor_item.get("visual_description") or "").strip(),
                    })

        if location_id:
            location_item = _item_for_id(self.manifest.get("locations") or [], location_id)
            if location_item:
                logical = _pick_existing_path(location_item.get("sheet_path"), self.project_dir)
                if logical:
                    images.append({
                        "path": logical,
                        "type": "location",
                        "id": location_id,
                        "visual_description": str(location_item.get("visual_description") or "").strip(),
                    })

        if not images:
            return {"sheet_path": "", "image_count": 0, "images": [], "scene_reference_sheet_description": ""}

        image_paths = [self.project_dir / img["path"] for img in images]
        shot_id = shot.get("shot_id") or f"scene_{shot.get('scene')}"
        output_path = self.project_dir / "movie" / "ingredients_sheets" / f"{shot_id}_ingredients.png"
        num_cols = math.ceil(math.sqrt(len(images)))
        compose_scene_reference_sheet(image_paths, output_path, size=self.size)

        relative_sheet = output_path.relative_to(self.project_dir).as_posix()
        description = generate_scene_sheet_description(images, num_cols, self.size)
        return {
            "sheet_path": relative_sheet,
            "image_count": len(images),
            "images": images,
            "scene_reference_sheet_description": description,
        }


def _pick_existing_path(value: object | None, project_dir: Path) -> str:
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
