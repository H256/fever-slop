from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrimSpec:
    source_file: Path
    output_file: Path
    fps: int
    trim_front_frames: int
    keep_frames: int
    scene: int

    @property
    def start_seconds(self) -> float:
        return self.trim_front_frames / float(self.fps)

    @property
    def duration_seconds(self) -> float:
        return self.keep_frames / float(self.fps)
