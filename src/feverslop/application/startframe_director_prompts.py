from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_startframe_director_prompts(*, project_dir: Path, candidate_count: int = 4) -> Path:
    project_dir = Path(project_dir)
    movie_dir = project_dir / "movie"
    plan = json.loads((movie_dir / "startframe_plan.json").read_text(encoding="utf-8"))
    identity = json.loads((movie_dir / "identity_ledger.json").read_text(encoding="utf-8"))
    shots = []
    for shot in plan.get("shots", []):
        prompt = _ideogram_prompt(shot, identity)
        shots.append(
            {
                "scene": int(shot.get("scene") or len(shots) + 1),
                "shot_id": str(shot.get("shot_id") or ""),
                "workflow": "workflows/image_t2i_startframe_ideogram_director_v1.json",
                "candidate_count": int(candidate_count),
                "positive_prompt": json.dumps(prompt, ensure_ascii=False),
                "negative_prompt": (
                    "extra people, duplicate characters, readable text, watermark, logo, "
                    "captions, malformed hands, wrong wardrobe, wrong location, split screen, "
                    "contact sheet, comic panels, multiple panels, storyboard sheet, reference sheet"
                ),
                "width": int(shot.get("width") or 1280),
                "height": int(shot.get("height") or 704),
            }
        )
    output_path = movie_dir / "startframe_director_prompts.json"
    output_path.write_text(json.dumps({"version": 1, "shots": shots}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def _ideogram_prompt(shot: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    actors = identity.get("actors") or {}
    elements = []
    for actor in shot.get("actors") or []:
        actor_id = str(actor.get("actor_id") or "")
        contract = actors.get(actor_id) or {}
        wardrobe = (contract.get("wardrobe") or {}).get("description") or ""
        elements.append(
            {
                "type": "obj",
                "actor_id": actor_id,
                "bbox": list(actor.get("bbox") or []),
                "desc": " ".join(
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
    return {
        "high_level_description": str((shot.get("startframe_intent") or {}).get("action_moment") or ""),
        "style_description": {
            "aesthetics": "cinematic film still, realistic production lighting",
            "lighting": str((shot.get("lighting") or {}).get("description") or ""),
            "medium": "LTX video startframe",
        },
        "compositional_deconstruction": {
            "camera": str((shot.get("startframe_intent") or {}).get("camera") or ""),
            "background": str((shot.get("continuity_in") or {}).get("location_id") or ""),
            "elements": elements,
            "props": list(shot.get("props") or []),
        },
        "continuity_constraints": {
            "must_show": list((shot.get("startframe_intent") or {}).get("must_show") or []),
            "must_not_show": list((shot.get("startframe_intent") or {}).get("must_not_show") or []),
            "required_carryovers": list((shot.get("continuity_in") or {}).get("required_carryovers") or []),
        },
    }
