from __future__ import annotations

from collections.abc import Callable, Sequence
import json
import math
import os
from pathlib import Path
from typing import Any


ProgressCallback = Callable[[int, int, str], None]


def export_render_plan_to_openshot(
    *,
    render_plan_path: str | Path,
    clip_paths: Sequence[str | Path],
    output_path: str | Path,
    width: int,
    height: int,
    fps: int,
    audio_path: str | Path | None = None,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Write an OpenShot .osp project whose clips follow the render plan timeline."""
    plan = json.loads(Path(render_plan_path).read_text(encoding="utf-8-sig"))
    if not isinstance(plan, list):
        raise ValueError(f"Render plan must be a JSON list: {render_plan_path}")
    if len(plan) != len(clip_paths):
        raise ValueError(
            "OpenShot export requires one rendered clip per render-plan entry "
            f"(got {len(clip_paths)} clips for {len(plan)} entries)"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    clips: list[dict[str, Any]] = []
    total_duration = 0.0
    total_items = len(plan) + (1 if audio_path is not None else 0)

    for index, (entry, clip_path) in enumerate(zip(plan, clip_paths, strict=True), start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Render plan entry {index} must be an object")
        scene_number = int(entry["scene"])
        duration = float(entry.get("duration_seconds", 0.0))
        if duration <= 0:
            raise ValueError(f"Render plan scene {scene_number} has no positive duration")
        position = float(entry.get("abs_start_seconds", total_duration))
        path = Path(clip_path)
        if not path.is_file():
            raise FileNotFoundError(f"Rendered clip does not exist: {path}")
        file_id = f"file_video_{scene_number:04}"
        files.append(_file_entry(file_id, path, output, media_type="video", duration=duration, width=width, height=height, fps=fps))
        clips.append(_clip_entry(
            f"clip_video_{scene_number:04}", file_id, position, duration, layer=1000000,
        ))
        total_duration = max(total_duration, position + duration)
        if on_progress:
            on_progress(index, total_items, f"scene {scene_number}")

    if audio_path is not None:
        audio = Path(audio_path)
        if not audio.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio}")
        audio_id = "file_audio_original"
        files.append(_file_entry(audio_id, audio, output, media_type="audio", duration=total_duration))
        clips.append(_clip_entry("clip_audio_original", audio_id, 0.0, total_duration, layer=2000000))
        if on_progress:
            on_progress(total_items, total_items, "audio")

    project = {
        "id": "T0",
        "fps": {"num": int(fps), "den": 1},
        "display_ratio": _ratio(width, height),
        "pixel_ratio": {"num": 1, "den": 1},
        "width": int(width),
        "height": int(height),
        "sample_rate": 48000,
        "channels": 2,
        "channel_layout": 3,
        "settings": {},
        "clips": clips,
        "effects": [],
        "files": files,
        "duration": total_duration,
        "scale": 15.0,
        "tick_pixels": 100,
        "playhead_position": 0.0,
        "profile": f"Custom {width}x{height} {fps} fps",
        "export_settings": None,
        "layers": [
            {"id": "L1", "label": "Audio", "number": 2000000, "y": 0, "lock": False},
            {"id": "L2", "label": "Video", "number": 1000000, "y": 1, "lock": False},
        ],
        "markers": [],
        "progress": [],
        "history": {"undo": [], "redo": []},
        "version": {"openshot-qt": "0.0.0", "libopenshot": "0.0.0"},
    }
    output.write_text(json.dumps(project, indent=2) + "\n", encoding="utf-8")
    return output


def _ratio(width: int, height: int) -> dict[str, int]:
    divisor = math.gcd(int(width), int(height))
    return {"num": int(width) // divisor, "den": int(height) // divisor}


def _relative_path(path: Path, project_file: Path) -> str:
    return Path(os.path.relpath(path.resolve(), project_file.parent.resolve())).as_posix()


def _file_entry(
    file_id: str,
    path: Path,
    project_file: Path,
    *,
    media_type: str,
    duration: float,
    width: int = 0,
    height: int = 0,
    fps: int = 0,
) -> dict[str, Any]:
    return {
        "id": file_id,
        "path": _relative_path(path, project_file),
        "media_type": media_type,
        "type": media_type,
        "name": path.name,
        "duration": duration,
        "has_video": media_type == "video",
        "has_audio": media_type in {"video", "audio"},
        "width": width,
        "height": height,
        "fps": {"num": fps, "den": 1} if fps else {"num": 0, "den": 1},
    }


def _clip_entry(clip_id: str, file_id: str, position: float, duration: float, *, layer: int) -> dict[str, Any]:
    return {
        "id": clip_id,
        "title": clip_id,
        "file_id": file_id,
        "position": position,
        "start": 0.0,
        "end": duration,
        "layer": layer,
        "effects": [],
        "animations": [],
        "gravity": "center",
        "scale": 1.0,
        "rotation": 0.0,
        "shear_x": 0.0,
        "shear_y": 0.0,
        "alpha": {"Points": []},
        "volume": {"Points": []},
    }
