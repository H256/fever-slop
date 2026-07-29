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
    crf: int = 18,
    force: bool = True,
) -> Path:
    """Encode PNG crop frames into an MP4 video using FFmpeg.

    Args:
        frames_folder: Directory containing numbered PNG frames.
        fps: Frame rate for the output video.
        output_path: Destination MP4 path.
        ffmpeg_path: Path to the FFmpeg binary.
        crf: Constant Rate Factor (0=lossless, 51=worse). Default 18 for near-lossless.
        force: Overwrite output file if it exists (-y flag).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pattern = str(frames_folder / "crop_%06d.png")
    cmd = [
        ffmpeg_path,
        "-r", str(fps),
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", str(crf),
    ]
    if force:
        cmd.append("-y")
    cmd.append(str(output_path))

    logger.info("Encoding face crop MP4: %s", output_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr_msg = (result.stderr or "no stderr output").strip()
        logger.error("FFmpeg failed (code=%d) for %s: %s", result.returncode, output_path, stderr_msg)
        raise RuntimeError(
            f"FFmpeg encoding failed for {output_path} (exit {result.returncode}): {stderr_msg}"
        )

    # Verify output integrity
    if not output_path.exists():
        logger.error("FFmpeg produced no output file: %s", output_path)
        raise RuntimeError(f"FFmpeg produced no output file: {output_path}")
    if output_path.stat().st_size == 0:
        logger.error("FFmpeg produced empty output: %s", output_path)
        raise RuntimeError(f"FFmpeg produced empty output: {output_path}")

    return output_path
