from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from feverslop.utils.io import read_json, read_json_or_none


def build_startframe_identity_ledger(*, project_dir: Path) -> Path:
    project_dir = Path(project_dir)
    movie_dir = project_dir / "movie"
    bible = read_json(movie_dir / "bible.json")
    manifest = _read_json_if_exists(movie_dir / "references" / "manifest.json")
    manifest_actors = {
        str(actor.get("id")): actor
        for actor in manifest.get("actors", [])
        if isinstance(actor, dict) and str(actor.get("id") or "").strip()
    }
    actors = {}
    for actor in bible.get("actors", []):
        if not isinstance(actor, dict):
            continue
        actor_id = str(actor.get("id") or "").strip()
        if not actor_id:
            continue
        manifest_actor = manifest_actors.get(actor_id, {})
        description = str(
            actor.get("visual_description")
            or manifest_actor.get("visual_description")
            or actor.get("name")
            or actor_id,
        ).strip()
        reference_path = str(
            manifest_actor.get("msr_sheet_path")
            or manifest_actor.get("sheet_path")
            or f"movie/references/actors/{actor_id}/msr_sheet.png",
        ).strip()
        actors[actor_id] = {
            "actor_id": actor_id,
            "name": str(actor.get("name") or manifest_actor.get("name") or actor_id),
            "reference_paths": {
                "face": reference_path,
                "full_body": reference_path,
                "wardrobe": reference_path,
            },
            "face": {
                "description": description,
                "must_preserve": _identity_terms(description),
                "may_change": ["sweat", "dirt", "minor blood smear when continuity allows"],
            },
            "body": {
                "description": description,
                "must_preserve": _identity_terms(description),
            },
            "wardrobe": {
                "description": description,
                "must_preserve": _identity_terms(description),
                "colors": _color_terms(description),
                "materials": _material_terms(description),
                "may_change": ["dust", "wrinkles", "wet fabric", "minor tears"],
                "forbidden": ["modern jacket", "white sneakers", "bright red shirt"],
            },
            "accessories": {
                "must_preserve": [],
                "may_appear": [],
            },
        }
    output = {"version": 1, "actors": actors}
    output_path = movie_dir / "identity_ledger.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    data = read_json_or_none(path)
    return {} if data is None else data


def _identity_terms(description: str) -> list[str]:
    text = " ".join(str(description or "").split())
    return [text] if text else []


def _color_terms(description: str) -> list[str]:
    colors = ("black", "white", "grey", "gray", "charcoal", "brown", "blue", "red", "green", "gold", "silver")
    lower = description.lower()
    return [color for color in colors if color in lower]


def _material_terms(description: str) -> list[str]:
    materials = ("wool", "cotton", "leather", "linen", "silk", "denim", "metal")
    lower = description.lower()
    return [material for material in materials if material in lower]

