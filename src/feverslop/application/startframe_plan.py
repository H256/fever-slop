from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_startframe_plan(*, project_dir: Path) -> Path:
    project_dir = Path(project_dir)
    movie_dir = project_dir / "movie"
    render_plan = _read_json(movie_dir / "render_plan.json")
    bible = _read_json(movie_dir / "bible.json")
    continuity = _read_json_if_exists(movie_dir / "continuity_plan.json")
    identity = _read_json(movie_dir / "identity_ledger.json")
    width, height = _resolution(render_plan)
    locations = {str(item.get("id")): item for item in bible.get("locations", []) if isinstance(item, dict)}
    shots = []
    for index, shot in enumerate(_shots(render_plan), start=1):
        shot_id = str(shot.get("shot_id") or f"shot_{index:04}")
        actor_ids = _actor_ids(shot)
        location_id = str(shot.get("location_id") or (shot.get("reference_ids") or {}).get("location") or "").strip()
        action = str(shot.get("action") or shot.get("description") or "").strip()
        carryovers = _carryovers(continuity, shot_id, location_id)
        actors = [
            _planned_actor(
                actor_id,
                index=actor_index,
                total=len(actor_ids),
                width=width,
                height=height,
                identity=identity,
            )
            for actor_index, actor_id in enumerate(actor_ids, start=1)
        ]
        shots.append(
            {
                "scene": int(shot.get("scene") or index),
                "shot_id": shot_id,
                "duration_seconds": float(shot.get("duration_seconds") or 4.0),
                "width": width,
                "height": height,
                "continuity_in": {
                    "location_id": location_id,
                    "story_state_before": " ".join(_continuity_list(continuity, shot_id, "incoming")),
                    "required_carryovers": carryovers,
                },
                "startframe_intent": {
                    "action_moment": action,
                    "camera": str(shot.get("camera") or "medium shot"),
                    "emotion": str(shot.get("expression") or "story-consistent emotion"),
                    "must_show": [action, *actor_ids, location_id],
                    "must_not_show": ["extra people", "duplicate characters", "readable text", "watermark"],
                },
                "actors": actors,
                "props": [],
                "lighting": {
                    "description": str((locations.get(location_id) or {}).get("visual_description") or shot.get("location") or ""),
                    "must_preserve": [],
                },
                "ltx_motion": {
                    "prompt": " ".join(
                        part
                        for part in (
                            "Use the supplied startframe as authoritative.",
                            action,
                            "Do not add people, change identity, change wardrobe, or change location.",
                        )
                        if part
                    ),
                    "allowed_motion": ["subtle story-consistent motion"],
                    "forbidden_motion": ["new characters enter", "location changes", "wardrobe changes"],
                },
            }
        )
    output = {"version": 1, "title": str(render_plan.get("title") or bible.get("title") or project_dir.name), "shots": shots}
    output_path = movie_dir / "startframe_plan.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path)


def _shots(render_plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [shot for shot in render_plan.get("shots") or render_plan.get("scenes") or [] if isinstance(shot, dict)]


def _resolution(render_plan: dict[str, Any]) -> tuple[int, int]:
    resolution = render_plan.get("resolution") or {}
    return int(resolution.get("width") or 1280), int(resolution.get("height") or 704)


def _actor_ids(shot: dict[str, Any]) -> list[str]:
    raw = shot.get("actor_ids") or (shot.get("reference_ids") or {}).get("actors") or []
    return [str(actor_id).strip() for actor_id in raw if str(actor_id).strip()]


def _planned_actor(actor_id: str, *, index: int, total: int, width: int, height: int, identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "actor_id": actor_id,
        "bbox": _actor_bbox(index=index, total=total, width=width, height=height),
        "screen_region": _screen_region(index=index, total=total),
        "pose": "story action pose",
        "depth_order": index,
        "visible_identity_parts": ["face", "hair", "wardrobe"],
        "repair_scopes": ["face_head", "torso_wardrobe"],
        "identity_contract": f"movie/identity_ledger.json#actors.{actor_id}",
        "wardrobe": ((identity.get("actors") or {}).get(actor_id) or {}).get("wardrobe", {}),
    }


def _actor_bbox(*, index: int, total: int, width: int, height: int) -> list[int]:
    top = int(height * 0.15)
    bottom = int(height * 0.95)
    if total <= 1:
        return [int(width * 0.30), top, int(width * 0.70), bottom]
    slot_width = width / total
    left = int(slot_width * (index - 1) + slot_width * 0.12)
    right = int(slot_width * index - slot_width * 0.12)
    return [left, top, right, bottom]


def _screen_region(*, index: int, total: int) -> str:
    if total <= 1:
        return "center foreground"
    if index == 1:
        return "left foreground"
    if index == total:
        return "right midground"
    return "center midground"


def _carryovers(continuity: dict[str, Any], shot_id: str, location_id: str) -> list[str]:
    carryovers = _continuity_list(continuity, shot_id, "required_carryovers")
    if location_id:
        carryovers.append(f"same {location_id} location")
    return list(dict.fromkeys(item for item in carryovers if item))


def _continuity_list(continuity: dict[str, Any], shot_id: str, key: str) -> list[str]:
    scene = (continuity.get("scene_continuity") or {}).get(shot_id) or {}
    value = scene.get(key) or []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if str(value).strip():
        return [str(value).strip()]
    return []
