from __future__ import annotations

import json
from pathlib import Path


def write_local_startframe_validation(*, project_dir: Path) -> Path:
    project_dir = Path(project_dir)
    plan = json.loads((project_dir / "movie" / "startframe_plan.json").read_text(encoding="utf-8"))
    shots = []
    for shot in plan.get("shots", []):
        scene_number = int(shot.get("scene") or len(shots) + 1)
        shots.append(
            {
                "scene": scene_number,
                "shot_id": str(shot.get("shot_id") or ""),
                "final_path": f"output/movie/storyboard/final/scene_{scene_number:04}.png",
                "pass": True,
                "scores": {
                    "character_presence": 1.0,
                    "action_state": 1.0,
                    "location_continuity": 1.0,
                    "wardrobe_identity": 1.0,
                    "face_identity_semantic": 1.0,
                    "prop_continuity": 1.0,
                    "ltx_startframe_usability": 1.0,
                },
                "failures": [],
                "warnings": ["local placeholder validation; real Gemma4 validation not run"],
                "validator": {"semantic_model": "local-placeholder", "threshold_profile": "default"},
            }
        )
    output_path = project_dir / "movie" / "startframe_validation.json"
    output_path.write_text(json.dumps({"version": 1, "shots": shots}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path

