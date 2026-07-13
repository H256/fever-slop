from __future__ import annotations

from typing import Any


def validate_project_config(data: Any, *, project_type: str = "standard_music_video") -> None:
    if not isinstance(data, dict):
        raise ValueError("config.json must be a JSON object")
    if not str(data.get("project_name") or "").strip():
        raise ValueError("project_name is required")
    if project_type != "movie" and not str(data.get("input_audio") or "").strip():
        raise ValueError("input_audio is required")
    if data.get("silent_mode") is not None and not isinstance(data["silent_mode"], bool):
        raise ValueError("silent_mode must be a boolean")
    subject_mode = str(data.get("subject_mode", "multi") or "multi").strip().lower()
    if subject_mode not in {"single", "multi"}:
        raise ValueError("subject_mode must be 'single' or 'multi'")
    video_pipeline = data.get("video_pipeline")
    if video_pipeline not in {None, "", "ltx_i2v", "ltx_msr", "ltx_ingredients"}:
        raise ValueError("video_pipeline must be 'ltx_i2v', 'ltx_msr', or 'ltx_ingredients'")
    max_scene_actors = int(data.get("max_scene_actors", 1 if subject_mode == "single" else 4))
    if max_scene_actors < 1 or max_scene_actors > 4:
        raise ValueError("max_scene_actors must be between 1 and 4")


def validate_full_auto_inputs(request: Any) -> None:
    if float(request.duration_seconds) <= 0:
        raise ValueError("duration_seconds must be positive")
    if int(request.width) <= 0:
        raise ValueError("width must be a positive integer")
    if int(request.height) <= 0:
        raise ValueError("height must be a positive integer")
    if int(request.fps) not in {16, 24, 50}:
        raise ValueError("fps must be one of 16, 24, or 50")
    if str(request.pipeline_mode or "classic") not in {"classic", "msr", "ingredients"}:
        raise ValueError("pipeline_mode must be classic, msr, or ingredients")
