from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from feverslop.config.project_config import ProjectConfig
from feverslop.utils.io import read_json_document, write_json_document


def build_startframe_director_prompts(
    *,
    project_dir: Path,
    candidate_count: int = 4,
    director_backend: str = "krea2",
    reference_image_size: tuple[int, int] | None = None,
) -> Path:
    project_dir = Path(project_dir)
    if reference_image_size is None and (project_dir / "config.json").is_file():
        project_config = ProjectConfig.load(project_dir / "config.json")
        reference_image_size = project_config.reference_images.resolve(project_config.video)
    movie_dir = project_dir / "movie"
    plan = _require_json(movie_dir / "startframe_plan.json", "startframe plan")
    identity = _require_json(movie_dir / "identity_ledger.json", "identity ledger")
    backend = _director_backend(director_backend)
    shots = []
    for shot in plan.get("shots", []):
        reference_width = int(reference_image_size[0]) if reference_image_size else int(shot.get("width") or 1280)
        reference_height = int(reference_image_size[1]) if reference_image_size else int(shot.get("height") or 704)
        positive_prompt = _positive_prompt(shot, identity, backend=backend)
        shots.append(
            {
                "scene": int(shot.get("scene") or len(shots) + 1),
                "shot_id": str(shot.get("shot_id") or ""),
                "director_backend": backend,
                "workflow": _director_workflow(backend),
                "candidate_count": int(candidate_count),
                "positive_prompt": positive_prompt,
                "negative_prompt": (
                    "extra people, duplicate characters, readable text, watermark, logo, "
                    "captions, malformed hands, wrong wardrobe, wrong location, split screen, "
                    "contact sheet, comic panels, multiple panels, storyboard sheet, reference sheet"
                ),
                "width": reference_width,
                "height": reference_height,
            }
        )
    output_path = movie_dir / "startframe_director_prompts.json"
    write_json_document(output_path, {"version": 1, "shots": shots})
    return output_path


def _require_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise ValueError(f"Missing required {label}: {path}")
    return read_json_document(path)


def _director_backend(value: str) -> str:
    backend = str(value or "krea2").strip().lower()
    if backend not in {"krea2", "ideogram"}:
        raise ValueError("director_backend must be krea2 or ideogram")
    return backend


def _director_workflow(backend: str) -> str:
    if backend == "ideogram":
        return "workflows/image_t2i_startframe_ideogram_director_v1.json"
    return "workflows/image_t2i_startframe_krea_v1.json"


def _positive_prompt(shot: dict[str, Any], identity: dict[str, Any], *, backend: str) -> str:
    if backend == "ideogram":
        return json.dumps(_ideogram_prompt(shot, identity), ensure_ascii=False)
    return _krea2_prompt(shot, identity)


def _ideogram_prompt(shot: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    actors = identity.get("actors") or {}
    objects = []
    for actor in shot.get("actors") or []:
        actor_id = str(actor.get("actor_id") or "")
        contract = actors.get(actor_id) or {}
        wardrobe = (contract.get("wardrobe") or {}).get("description") or ""
        objects.append(
            {
                "label": str(contract.get("name") or actor_id),
                "kind": "person",
                "bounding_box": list(actor.get("bbox") or []),
                "description": " ".join(
                    part
                    for part in (
                        str(contract.get("name") or actor_id),
                        str((contract.get("face") or {}).get("description") or ""),
                        str(wardrobe),
                        str(actor.get("pose") or ""),
                    )
                    if part
                ),
            }
        )
    location = str((shot.get("continuity_in") or {}).get("location_id") or "")
    return {
        "scene_summary": str((shot.get("startframe_intent") or {}).get("action_moment") or ""),
        "style": {
            "medium": "single cinematic film still, realistic production lighting",
            "lighting": str((shot.get("lighting") or {}).get("description") or ""),
            "avoid": "text overlays, contact sheets, split screens, multiple panels",
        },
        "background": {
            "location": location,
            "camera": str((shot.get("startframe_intent") or {}).get("camera") or ""),
        },
        "objects": objects,
        "props": list(shot.get("props") or []),
    }


def _krea2_prompt(shot: dict[str, Any], identity: dict[str, Any]) -> str:
    actors = identity.get("actors") or {}
    actor_parts = []
    for actor in shot.get("actors") or []:
        actor_id = str(actor.get("actor_id") or "")
        contract = actors.get(actor_id) or {}
        wardrobe = (contract.get("wardrobe") or {}).get("description") or ""
        face = (contract.get("face") or {}).get("description") or ""
        pose = str(actor.get("pose") or "").strip()
        bbox = actor.get("bbox") or []
        placement = f"placed in frame region {bbox}" if bbox else "placed clearly in the frame"
        actor_parts.append(
            " ".join(
                part
                for part in (
                    str(contract.get("name") or actor_id),
                    face,
                    f"wearing {wardrobe}" if wardrobe else "",
                    pose,
                    placement,
                )
                if part
            )
        )
    intent = shot.get("startframe_intent") or {}
    camera = str(intent.get("camera") or shot.get("camera") or "").strip()
    action = str(intent.get("action_moment") or shot.get("action") or shot.get("description") or "").strip()
    location = str((shot.get("continuity_in") or {}).get("location_id") or shot.get("location") or "").strip()
    lighting = str((shot.get("lighting") or {}).get("description") or "").strip()
    props = ", ".join(str(prop) for prop in shot.get("props") or [] if prop)
    parts = [
        "A single cinematic film still for a movie start frame.",
        action,
        f"Camera: {camera}." if camera else "",
        f"Location: {location}." if location else "",
        f"Lighting: {lighting}." if lighting else "",
        "Characters: " + " | ".join(actor_parts) + "." if actor_parts else "",
        f"Visible props: {props}." if props else "",
        "Realistic production design, natural human proportions, coherent hands, no text, no split screen, no contact sheet, no storyboard grid.",
    ]
    return " ".join(part for part in parts if part)
