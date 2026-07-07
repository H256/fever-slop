from __future__ import annotations

import json
from pathlib import Path


def write_startframe_i2v_render_plan(*, project_dir: Path, fps: int = 24) -> Path:
    project_dir = Path(project_dir)
    plan = json.loads((project_dir / "movie" / "startframe_plan.json").read_text(encoding="utf-8"))
    scenes = []
    cursor = 0.0
    for shot in plan.get("shots", []):
        duration = float(shot.get("duration_seconds") or 4.0)
        scene_number = int(shot.get("scene") or len(scenes) + 1)
        video_prompt = str((shot.get("ltx_motion") or {}).get("prompt") or "")
        scenes.append(
            {
                "scene": scene_number,
                "duration_seconds": duration,
                "abs_start_seconds": cursor,
                "fps": fps,
                "width": int(shot.get("width") or 1280),
                "height": int(shot.get("height") or 704),
                "frame_count": max(1, round(duration * fps)),
                "z_image": {"prompt": str((shot.get("startframe_intent") or {}).get("action_moment") or "")},
                "ltx": {
                    "original_style_i2v_prompt": video_prompt,
                    "i2v_prompt_from_t2i": video_prompt,
                },
                "movie": {
                    "shot_id": str(shot.get("shot_id") or ""),
                    "startframe_source": f"output/movie/storyboard/final/scene_{scene_number:04}.png",
                    "startframe_validation_required": True,
                },
            }
        )
        cursor += duration
    output_path = project_dir / "movie" / "render_plan_i2v.json"
    output_path.write_text(json.dumps(scenes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path

