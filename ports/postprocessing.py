from __future__ import annotations

from pathlib import Path
from typing import Protocol

from video_postprocessor import TrimSpec


class PostProcessorPort(Protocol):
    def trim_clip(self, spec: TrimSpec) -> Path:
        """Trim a rendered clip to its scene window."""

    def write_concat_list(self, video_files: list[Path], output_file: str | Path) -> Path:
        """Write an ffmpeg concat list."""

    def mux_original_audio(
        self,
        video_file: str | Path,
        audio_file: str | Path,
        output_file: str | Path,
    ) -> Path:
        """Mux a final video with the original song audio."""
