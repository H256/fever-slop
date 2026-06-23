from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from feverslop.domain.full_auto import SongSpec
from feverslop.ports.full_auto import (
    PipelineRunnerPort,
    ProjectScaffoldPort,
    SongAudioGeneratorPort,
    SongBriefGeneratorPort,
)


@dataclass(frozen=True)
class FullAutoRequest:
    idea: str
    style: str
    project_name: str | None = None
    projects_dir: Path = Path("projects")
    duration_seconds: float = 120.0
    language: str = "en"
    bpm: int | None = None
    keyscale: str | None = None
    seed: int = 0
    run_video_pipeline: bool = False
    runner_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FullAutoResult:
    project_dir: Path
    project_config_path: Path
    audio_path: Path
    lyrics_path: Path
    song_spec_path: Path
    song_spec: SongSpec
    song_manifest: dict[str, Any]
    final_video_path: Path | None = None


class FullAutoUseCase:
    def __init__(
        self,
        *,
        brief_generator: SongBriefGeneratorPort,
        song_generator: SongAudioGeneratorPort,
        project_scaffold: ProjectScaffoldPort,
        pipeline_runner: PipelineRunnerPort | None = None,
    ):
        self.brief_generator = brief_generator
        self.song_generator = song_generator
        self.project_scaffold = project_scaffold
        self.pipeline_runner = pipeline_runner

    def execute(self, request: FullAutoRequest) -> FullAutoResult:
        project_slug = slugify_project_name(request.project_name or request.idea)
        spec = self._apply_overrides(self.brief_generator.generate(request), request)
        generated_song = self.song_generator.generate(
            spec,
            project_slug=project_slug,
            output_dir=Path(request.projects_dir) / project_slug / "input",
            seed=int(request.seed),
        )
        scaffold = self.project_scaffold.create_project(
            projects_dir=Path(request.projects_dir),
            project_slug=project_slug,
            spec=spec,
            generated_song=generated_song,
        )

        final_video_path = None
        if request.run_video_pipeline:
            if self.pipeline_runner is None:
                raise ValueError("FullAutoUseCase requires a pipeline_runner when run_video_pipeline is true")
            final_video_path = self.pipeline_runner.run(
                project_config_path=scaffold.project_config_path,
                options=dict(request.runner_options),
            )

        return FullAutoResult(
            project_dir=scaffold.project_dir,
            project_config_path=scaffold.project_config_path,
            audio_path=scaffold.audio_path,
            lyrics_path=scaffold.lyrics_path,
            song_spec_path=scaffold.song_spec_path,
            song_spec=spec,
            song_manifest=generated_song.manifest,
            final_video_path=final_video_path,
        )

    @staticmethod
    def _apply_overrides(spec: SongSpec, request: FullAutoRequest) -> SongSpec:
        return SongSpec(
            title=spec.title,
            tags=spec.tags,
            lyrics=spec.lyrics,
            bpm=int(request.bpm) if request.bpm is not None else int(spec.bpm),
            duration_seconds=float(request.duration_seconds),
            language=str(request.language or spec.language),
            keyscale=str(request.keyscale or spec.keyscale),
            visual_story_idea=spec.visual_story_idea,
            visual_style=spec.visual_style,
        )


def slugify_project_name(value: str) -> str:
    import re

    raw = str(value or "").strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    return safe or "full_auto_song"
