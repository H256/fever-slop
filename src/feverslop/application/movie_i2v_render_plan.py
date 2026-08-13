from __future__ import annotations

import json
from pathlib import Path


def write_movie_i2v_render_plan(
    *,
    project_dir: Path,
    fps: int = 24,
    width: int = 1280,
    height: int = 704,
) -> Path:
    project_dir = Path(project_dir)
    visual_plan_path = project_dir / "movie" / "visual_plan.json"
    if not visual_plan_path.is_file():
        raise FileNotFoundError(f"Movie visual plan not found: {visual_plan_path}")
    visual_plan = json.loads(visual_plan_path.read_text(encoding="utf-8"))
    scenes = []
    cursor = 0.0
    for shot in visual_plan.get("shots", []):
        duration = float(shot.get("duration_seconds") or 4.0)
        frame_count = max(1, round(duration * fps))
        video_prompt = str(shot.get("video_prompt") or "")
        scenes.append({
            "scene": int(shot.get("scene") or len(scenes) + 1),
            "duration_seconds": duration,
            "abs_start_seconds": cursor,
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "z_image": {"prompt": str(shot.get("base_plate_prompt") or "")},
            "ltx": {
                "original_style_i2v_prompt": video_prompt,
                "i2v_prompt_from_t2i": video_prompt,
            },
            "movie": {
                "shot_id": str(shot.get("shot_id") or ""),
                "view_id": str(shot.get("view_id") or ""),
                "edit_passes": list(shot.get("edit_passes") or []),
                "selected_actor_ids": list(shot.get("selected_actor_ids") or []),
            },
        })
        cursor += duration
    output_path = project_dir / "movie" / "render_plan_i2v.json"
    output_path.write_text(json.dumps(scenes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path
