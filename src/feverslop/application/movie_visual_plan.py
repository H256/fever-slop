from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_movie_visual_plan(*, project_dir: Path) -> Path:
    project_dir = Path(project_dir)
    movie_dir = project_dir / "movie"
    render_plan = _read_json(movie_dir / "render_plan.json")
    bible = _read_json(movie_dir / "bible.json")
    manifest = _read_json_if_exists(movie_dir / "references" / "manifest.json")
    locations = {str(item.get("id")): item for item in bible.get("locations", [])}
    actors = {str(item.get("id")): item for item in bible.get("actors", [])}
    actor_reference_paths = _actor_reference_paths(manifest)
    shots = _movie_shots(render_plan)
    scene_views = _derive_scene_views(shots, locations)
    visual_shots = [
        _visual_shot(shot, actors=actors, locations=locations, actor_reference_paths=actor_reference_paths)
        for shot in shots
    ]
    output = {
        "title": render_plan.get("title") or bible.get("title") or project_dir.name,
        "version": 1,
        "scene_views": scene_views,
        "shots": visual_shots,
    }
    output_path = movie_dir / "visual_plan.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path)


def _movie_shots(render_plan: dict[str, Any]) -> list[dict[str, Any]]:
    return list(render_plan.get("shots") or render_plan.get("scenes") or [])


def _actor_reference_paths(manifest: dict[str, Any]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for actor in manifest.get("actors") or []:
        if not isinstance(actor, dict):
            continue
        actor_id = str(actor.get("id") or "").strip()
        reference_path = str(actor.get("msr_sheet_path") or actor.get("sheet_path") or "").strip()
        if actor_id and reference_path:
            paths[actor_id] = reference_path
    return paths


def _derive_scene_views(shots: list[dict[str, Any]], locations: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    views = []
    seen: set[str] = set()
    for shot in shots:
        location_id = _shot_location_id(shot)
        framing = _framing_bucket(str(shot.get("camera") or shot.get("description") or ""))
        view_id = f"{location_id}_{framing}"
        if view_id in seen:
            continue
        seen.add(view_id)
        location = locations.get(location_id, {})
        views.append({
            "view_id": view_id,
            "location_id": location_id,
            "name": f"{location.get('name') or shot.get('location') or location_id} {framing}",
            "framing": framing,
            "description": location.get("visual_description") or shot.get("location") or location_id,
            "base_plate_path": "",
        })
    return views


def _visual_shot(
    shot: dict[str, Any],
    *,
    actors: dict[str, dict[str, Any]],
    locations: dict[str, dict[str, Any]],
    actor_reference_paths: dict[str, str],
) -> dict[str, Any]:
    location_id = _shot_location_id(shot)
    framing = _framing_bucket(str(shot.get("camera") or shot.get("description") or ""))
    actor_ids = [str(item) for item in shot.get("actor_ids", []) if str(item).strip()]
    return {
        "shot_id": str(shot.get("shot_id") or f"shot_{int(shot.get('scene') or 1):04}"),
        "scene": int(shot.get("scene") or 1),
        "duration_seconds": float(shot.get("duration_seconds") or 4.0),
        "view_id": f"{location_id}_{framing}",
        "selected_actor_ids": actor_ids,
        "base_plate_prompt": _base_plate_prompt(shot, locations.get(location_id, {}), framing),
        "edit_passes": _edit_passes(actor_ids, actors, actor_reference_paths),
        "video_prompt": _video_prompt(shot),
    }


def _shot_location_id(shot: dict[str, Any]) -> str:
    return str(shot.get("location_id") or _slug(shot.get("location") or "location"))


def _base_plate_prompt(shot: dict[str, Any], location: dict[str, Any], framing: str) -> str:
    setting = location.get("visual_description") or shot.get("location") or "the scene location"
    return " ".join([
        "Scene-only background plate.",
        f"Framing: {framing}.",
        f"Setting: {setting}.",
        str(shot.get("action") or shot.get("description") or "").strip(),
        "Do not render people, faces, bodies, silhouettes, captions, labels, logos, or readable text.",
    ]).strip()


def _edit_passes(actor_ids: list[str], actors: dict[str, dict[str, Any]], actor_reference_paths: dict[str, str]) -> list[dict[str, Any]]:
    passes = []
    for index, actor_id in enumerate(actor_ids, start=1):
        placement_zone = _placement_zone(index)
        prompt = "\n".join([
            "Image 1 is the current scene plate.",
            f"Image 2 is the character reference image for {actor_id}.",
            f"Add only {actor_id} from Image 2 into Image 1 at natural scale.",
            f"Place {actor_id} in the {placement_zone}.",
            "The character's feet must touch the visible floor plane or ground plane, with believable contact shadows.",
            "Use full-body standing human scale relative to doors, hearths, furniture, shelves, jars, tools, and stones in Image 1.",
            "Do not place the character on shelves, baskets, barrels, pots, tools, furniture, walls, or inside containers.",
            "Do not make the character miniature, floating, sitting in props, embedded in objects, or cropped by nearby objects unless the shot action explicitly says so.",
            "Preserve the existing scene plate and every already-present character exactly.",
            "Preserve the identity, clothing, and natural proportions of the character from Image 2.",
            "Do not add extra people, duplicate characters, captions, labels, logos, watermarks, or readable text.",
        ])
        passes.append({
            "pass": index,
            "actor_id": actor_id,
            "actor_name": str(actors.get(actor_id, {}).get("name") or actor_id),
            "placement_zone": placement_zone,
            "input_plate_path": "",
            "reference_image_path": actor_reference_paths.get(actor_id, f"movie/references/actors/{actor_id}/hero.png"),
            "output_path": "",
            "prompt": prompt,
        })
    return passes


def _placement_zone(index: int) -> str:
    zones = [
        "center-right foreground floor area",
        "center-left foreground floor area",
        "center midground floor area",
        "right midground floor area",
        "left midground floor area",
    ]
    return zones[(max(1, index) - 1) % len(zones)]


def _video_prompt(shot: dict[str, Any]) -> str:
    action = str(shot.get("action") or shot.get("description") or "").strip()
    camera = str(shot.get("camera") or "").strip()
    dialogue = str(shot.get("dialogue") or "").strip()
    parts = [
        "Use the supplied startframe as the authoritative composition, identity, wardrobe, props, and environment.",
        action,
        camera,
        f"Spoken dialogue: {dialogue}" if dialogue else "",
        "Animate only the planned action with subtle cinematic motion; do not change location or add people.",
    ]
    return " ".join(part for part in parts if part).strip()


def _framing_bucket(value: str) -> str:
    lower = value.lower()
    if "close" in lower:
        return "close"
    if "wide" in lower or "establish" in lower:
        return "wide"
    if "insert" in lower or "detail" in lower:
        return "insert"
    return "medium"


def _slug(value: object) -> str:
    text = str(value or "").lower()
    slug = "".join(char if char.isalnum() else "_" for char in text).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "location"
