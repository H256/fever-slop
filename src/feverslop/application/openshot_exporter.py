from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import subprocess
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
    if isinstance(plan, dict):
        plan = plan.get("shots") or plan.get("scenes") or []
    if not isinstance(plan, list):
        raise ValueError(f"Render plan must be a JSON list: {render_plan_path}")
    if len(plan) != len(clip_paths):
        raise ValueError(
            "OpenShot export requires one rendered clip per render-plan entry "
            f"(got {len(clip_paths)} clips for {len(plan)} entries)"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    profile = _profile_from_clips(clip_paths) or (int(width), int(height), int(fps))
    width, height, fps = profile
    files: list[dict[str, Any]] = []
    clips: list[dict[str, Any]] = []
    total_duration = 0.0
    total_items = len(plan) + (1 if audio_path is not None else 0)

    for index, (entry, clip_path) in enumerate(zip(plan, clip_paths, strict=True), start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Render plan entry {index} must be an object")
        scene_number = int(entry.get("scene") or entry.get("scene_number") or index)
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
            f"clip_video_{scene_number:04}", file_id, position, duration,
            layer=1000000, reader_path=_relative_path(path, output),
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
        clips.append(_clip_entry(
            "clip_audio_original", audio_id, 0.0, total_duration,
            layer=2000000, reader_path=_relative_path(audio, output),
        ))
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
        # 0.0.0 is OpenShot's beta marker and makes current OpenShot Qt run
        # legacy migrations (including inverting alpha keyframes).
        "version": {"openshot-qt": "3.5.1", "libopenshot": "0.7.0"},
    }
    output.write_text(json.dumps(project, indent=2) + "\n", encoding="utf-8")
    return output


def _ratio(width: int, height: int) -> dict[str, int]:
    divisor = math.gcd(int(width), int(height))
    return {"num": int(width) // divisor, "den": int(height) // divisor}


def _profile_from_clips(clip_paths: Sequence[str | Path]) -> tuple[int, int, int] | None:
    detected: tuple[int, int, int] | None = None
    detected_path: Path | None = None
    for clip_path in clip_paths:
        profile = _probe_video_profile(Path(clip_path))
        if profile is None:
            continue
        if detected is None:
            detected = profile
            detected_path = Path(clip_path)
        elif profile != detected:
            raise ValueError(
                "OpenShot export requires matching rendered clip profiles: "
                f"{detected_path} has {detected[0]}x{detected[1]}@{detected[2]}fps, "
                f"but {clip_path} has {profile[0]}x{profile[1]}@{profile[2]}fps"
            )
    return detected


def _probe_video_profile(path: Path) -> tuple[int, int, int] | None:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate",
                "-of", "json", str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        streams = json.loads(result.stdout).get("streams") or []
        stream = streams[0]
        width = int(stream["width"])
        height = int(stream["height"])
        rate = Fraction(str(stream["r_frame_rate"]))
        fps = round(float(rate))
        if width <= 0 or height <= 0 or fps <= 0:
            return None
        return width, height, fps
    except (OSError, IndexError, KeyError, TypeError, ValueError, subprocess.CalledProcessError):
        return None


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


def _clip_entry(
    clip_id: str,
    file_id: str,
    position: float,
    duration: float,
    *,
    layer: int,
    reader_path: str,
) -> dict[str, Any]:
    return {
        "id": clip_id,
        "title": clip_id,
        "file_id": file_id,
        "reader": {"type": "FFmpegReader", "path": reader_path},
        "position": position,
        "start": 0.0,
        "end": duration,
        "layer": layer,
        "effects": [],
        "animations": [],
        # libopenshot stores these as enum values in the timeline JSON.  In
        # particular, passing the UI label "center" makes Timeline::SetJson
        # fail when it reads the value with JsonCpp::asInt().
        "gravity": 4,  # GRAVITY_CENTER
        "scale": 1,  # SCALE_FIT
        # OpenShot Qt upgrades these two fields by iterating Points, so they
        # need a real initial point rather than an empty keyframe object.
        "rotation": 0.0,
        "shear_x": 0.0,
        "shear_y": 0.0,
        "alpha": _keyframe(1.0),
        "volume": _keyframe(1.0),
    }


def _keyframe(value: float) -> dict[str, list[dict[str, Any]]]:
    """Return a constant libopenshot keyframe with one initial point."""
    return {
        "Points": [{
            "co": {"X": 1.0, "Y": float(value)},
            "interpolation": 0,
        }],
    }
