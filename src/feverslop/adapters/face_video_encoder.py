from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def encode_face_crop_mp4(
    frames_folder: Path,
    fps: float,
    output_path: Path,
    ffmpeg_path: str = "ffmpeg",
) -> Path:
    """Encode PNG crop frames into an MP4 video using FFmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pattern = str(frames_folder / "crop_%06d.png")
    cmd = [
        ffmpeg_path,
        "-r", str(fps),
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "18",
        "-y",
        str(output_path),
    ]

    logger.info("Encoding face crop MP4: %s", output_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("FFmpeg failed: %s", result.stderr)
        raise RuntimeError(f"FFmpeg encoding failed: {result.stderr}")

    return output_path
