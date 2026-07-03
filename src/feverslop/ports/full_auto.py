from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from feverslop.domain.full_auto import GeneratedSong, ProjectScaffoldResult, SongSpec


class SongBriefGeneratorPort(Protocol):
    def generate(self, request: Any) -> SongSpec:
        """Generate a structured song and visual brief from a FullAutoRequest."""


class SongAudioGeneratorPort(Protocol):
    def generate(
        self,
        spec: SongSpec,
        *,
        project_slug: str,
        output_dir: Path,
        seed: int,
    ) -> GeneratedSong:
        """Generate a song audio file for a structured song spec."""


class ProjectScaffoldPort(Protocol):
    def create_project(
        self,
        *,
        projects_dir: Path,
        project_slug: str,
        spec: SongSpec,
        generated_song: GeneratedSong,
        width: int,
        height: int,
        fps: int = 24,
        video_pipeline: str = "ltx_i2v",
    ) -> ProjectScaffoldResult:
        """Create a FeverSlop project folder and config from generated song assets."""


class PipelineRunnerPort(Protocol):
    def run(self, *, project_config_path: Path, options: dict[str, Any]) -> Path | None:
        """Run the downstream video pipeline and return the final video path when available."""
