from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Default wall-clock budget for a single FFmpeg operation (trim/re-encode,
# concat, mux, frame extraction). Slow scene re-encodes (libx264, CRF 18,
# preset slow) can exceed 120s, so the default is a practical 10 minutes and
# is overridable per project through the app config (see ComfyUIConfig).
FFMPEG_TIMEOUT_SECONDS = 600.0


@dataclass(frozen=True)
class TrimSpec:
    source_file: Path
    output_file: Path
    fps: int
    trim_front_frames: int
    keep_frames: int
    scene: int
    extract_boundary_frames: bool = False

    @property
    def start_seconds(self) -> float:
        return self.trim_front_frames / float(self.fps)

    @property
    def duration_seconds(self) -> float:
        return self.keep_frames / float(self.fps)
