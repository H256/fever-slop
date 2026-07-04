from __future__ import annotations

import json
from pathlib import Path

from feverslop.application.movie import build_movie_actor_reference_prompt


def ensure_movie_bible(project_dir: Path) -> Path:
    project_dir = Path(project_dir)
    bible_path = project_dir / "movie" / "bible.json"
    if bible_path.exists():
        return bible_path
    render_plan_path = project_dir / "movie" / "render_plan.json"
    if not render_plan_path.exists():
        raise FileNotFoundError(f"Movie render plan not found: {render_plan_path}")
    render_plan = _read_json(render_plan_path)
    manifest_path = project_dir / "movie" / "references" / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {"actors": [], "locations": []}
    bible = _legacy_bible_from_render_plan(render_plan, manifest, project_dir=project_dir)
    bible_path.write_text(json.dumps(bible, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return bible_path


def ensure_movie_render_plan_matches_bible(project_dir: Path) -> Path:
    render_plan_path = Path(project_dir) / "movie" / "render_plan.json"
    if not render_plan_path.exists():
        raise FileNotFoundError(f"Movie render plan not found: {render_plan_path}")
    return render_plan_path


def write_movie_reference_manifest_from_bible(project_dir: Path) -> Path:
    project_dir = Path(project_dir)
    bible_path = ensure_movie_bible(project_dir)
    manifest_path = project_dir / "movie" / "references" / "manifest.json"
    bible = _read_json(bible_path)
    existing = _read_json(manifest_path) if manifest_path.exists() else {}
    existing_actors = {str(actor.get("id")): actor for actor in existing.get("actors") or [] if isinstance(actor, dict)}
    existing_locations = {str(location.get("id")): location for location in existing.get("locations") or [] if isinstance(location, dict)}
    manifest = dict(existing)
    manifest["project_type"] = "movie"
    manifest["actors"] = [_manifest_actor(actor, existing_actors.get(str(actor.get("id"))) or {}) for actor in bible.get("actors") or [] if isinstance(actor, dict)]
    manifest["locations"] = [
        _manifest_location(location, existing_locations.get(str(location.get("id"))) or {}) for location in bible.get("locations") or [] if isinstance(location, dict)
    ]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def _manifest_actor(actor: dict, current: dict) -> dict:
    visual_description = str(actor.get("visual_description") or actor.get("name") or actor.get("id") or "").strip()
    prompt = build_movie_actor_reference_prompt(str(actor.get("name") or actor.get("id")), visual_description)
    return {
        **current,
        "id": actor.get("id"),
        "name": actor.get("name") or actor.get("id"),
        "role": actor.get("role") or "",
        "visual_description": visual_description,
        "image_prompt": prompt,
        "prompt": prompt,
        "status": current.get("status") or "required",
        "msr_sheet_path": current.get("msr_sheet_path") or "",
    }


def _manifest_location(location: dict, current: dict) -> dict:
    visual_description = str(location.get("visual_description") or location.get("name") or location.get("id") or "").strip()
    return {
        **current,
        "id": location.get("id"),
        "name": location.get("name") or location.get("id"),
        "visual_description": visual_description,
        "image_prompt": visual_description,
        "prompt": visual_description,
        "status": current.get("status") or "required",
        "msr_sheet_path": current.get("msr_sheet_path") or "",
    }


def _legacy_bible_from_render_plan(render_plan: dict, manifest: dict, *, project_dir: Path) -> dict:
    resolution = render_plan.get("resolution") or {}
    bible = {
        "title": render_plan.get("title") or project_dir.name,
        "premise": render_plan.get("premise") or "",
        "story_arch": {
            "title": render_plan.get("title") or project_dir.name,
            "premise": render_plan.get("premise") or "",
            "beats": [str(shot.get("description") or shot.get("action") or "") for shot in render_plan.get("shots") or [] if isinstance(shot, dict)],
        },
        "actors": [
            {
                "id": actor.get("id"),
                "name": actor.get("name") or actor.get("id"),
                "role": actor.get("role") or "",
                "visual_description": actor.get("visual_description") or actor.get("image_prompt") or actor.get("prompt") or actor.get("name") or actor.get("id"),
            }
            for actor in manifest.get("actors") or []
            if isinstance(actor, dict) and actor.get("id")
        ],
        "locations": [
            {
                "id": location.get("id"),
                "name": location.get("name") or location.get("id"),
                "visual_description": location.get("visual_description") or location.get("image_prompt") or location.get("prompt") or location.get("name") or location.get("id"),
            }
            for location in manifest.get("locations") or []
            if isinstance(location, dict) and location.get("id")
        ],
        "continuity": [{"id": "legacy_visual_continuity", "description": "Preserve existing actor and location references from the legacy movie manifest."}],
        "style_constraints": [],
        "runtime_constraints": {
            "width": int(resolution.get("width") or 1280),
            "height": int(resolution.get("height") or 704),
            "max_scene_actors": 4,
        },
    }
    if not bible["actors"]:
        bible["actors"] = [{"id": "main_character", "name": "Main Character", "role": "lead", "visual_description": "story-defined cinematic lead character"}]
    if not bible["locations"]:
        bible["locations"] = [{"id": "primary_location", "name": "Primary Location", "visual_description": "story-defined cinematic location"}]
    return bible


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Movie artifact must be a JSON object: {path}")
    return data
