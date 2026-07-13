from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from feverslop.domain.full_auto import SongSpec
from feverslop.domain.slug_utils import slugify_project_name
from feverslop.errors import FeverSlopConfigError
from feverslop.ports.full_auto import (
    PipelineRunnerPort,
    ProjectScaffoldPort,
    SongAudioGeneratorPort,
    SongBriefGeneratorPort,
)
from feverslop.ports.reporting import ConsoleReporter, NullReporter, Reporter


@dataclass(frozen=True)
class FullAutoRequest:
    idea: str
    style: str
    project_name: str | None = None
    projects_dir: Path = Path("projects")
    duration_seconds: float = 120.0
    width: int = 1280
    height: int = 704
    fps: int = 24
    language: str = "en"
    bpm: int | None = None
    keyscale: str | None = None
    seed: int = 0
    silent_mode: bool = False
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
        console: object | None = None,
        reporter: Reporter | None = None,
    ):
        self.brief_generator = brief_generator
        self.song_generator = song_generator
        self.project_scaffold = project_scaffold
        self.pipeline_runner = pipeline_runner
        if reporter is not None:
            self.reporter = reporter
        elif console is not None:
            self.reporter = ConsoleReporter(console)
        else:
            self.reporter = NullReporter()

    def execute(self, request: FullAutoRequest) -> FullAutoResult:
        project_slug = slugify_project_name(request.project_name or request.idea)
        self._print_startup(request=request, project_slug=project_slug)

        self.log_step("1. Generating ACE-Step song brief")
        spec = self._apply_overrides(self.brief_generator.generate(request), request)
        self._print_song_spec(spec)

        self.log_step("2. Rendering ACE-Step audio")
        generated_song = self.song_generator.generate(
            spec,
            project_slug=project_slug,
            output_dir=Path(request.projects_dir) / project_slug / "input",
            seed=int(request.seed),
        )
        self.log_file("Generated audio", generated_song.audio_path)

        self.log_step("3. Creating FeverSlop project")
        scaffold = self.project_scaffold.create_project(
            projects_dir=Path(request.projects_dir),
            project_slug=project_slug,
            spec=spec,
            generated_song=generated_song,
            width=int(request.width),
            height=int(request.height),
            fps=int(request.fps),
            video_pipeline=str(request.runner_options.get("video_pipeline") or "ltx_i2v"),
            silent_mode=bool(request.silent_mode),
        )
        self.log_file("Project config", scaffold.project_config_path)
        self.log_file("Lyrics", scaffold.lyrics_path)
        self.log_file("Song spec", scaffold.song_spec_path)

        final_video_path = None
        if request.run_video_pipeline:
            if self.pipeline_runner is None:
                raise FeverSlopConfigError("FullAutoUseCase requires a pipeline_runner when run_video_pipeline is true")
            self.log_step("4. Running video pipeline")
            final_video_path = self.pipeline_runner.run(
                project_config_path=scaffold.project_config_path,
                options=dict(request.runner_options),
            )
            if final_video_path:
                self.log_file("Final video", final_video_path)
        else:
            self.reporter.message("[yellow]Skipping video pipeline; project is prepared for later rendering.[/yellow]")

        self._print_complete(
            scaffold=scaffold,
            final_video_path=final_video_path,
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

    def log_step(self, title: str) -> None:
        self.reporter.step(title)

    def log_file(self, label: str, path: Path) -> None:
        self.reporter.file(label, path)

    def _print_startup(self, *, request: FullAutoRequest, project_slug: str) -> None:
        self.reporter.panel(
            f"[bold]Full-Auto ACE-Step Pipeline[/bold]\n\n"
            f"Project: [cyan]{project_slug}[/cyan]\n"
            f"Duration: [yellow]{float(request.duration_seconds):.1f}s[/yellow]\n"
            f"Resolution: [yellow]{int(request.width)}x{int(request.height)} @ {int(request.fps)}fps[/yellow]\n"
            f"Language: [yellow]{request.language}[/yellow]\n"
            f"Seed: [yellow]{int(request.seed)}[/yellow]\n"
            f"Video pipeline: [yellow]{'on' if request.run_video_pipeline else 'off'}[/yellow]",
            title="Startup",
        )

    def _print_song_spec(self, spec: SongSpec) -> None:
        self.reporter.table(
            "Generated Song Brief",
            ["Field", "Value"],
            [
                ["Title", spec.title],
                ["BPM", str(spec.bpm)],
                ["Duration", f"{float(spec.duration_seconds):.1f}s"],
                ["Language", spec.language],
                ["Key", spec.keyscale],
            ],
        )
        self.reporter.panel(spec.tags, title="ACE-Step Tags")
        self.reporter.panel(spec.visual_story_idea, title="Video Story")

    def _print_complete(self, *, scaffold, final_video_path: Path | None) -> None:
        lines = [
            "[bold green]Done.[/bold green]",
            "",
            f"Project config: [cyan]{scaffold.project_config_path}[/cyan]",
            f"Audio: [cyan]{scaffold.audio_path}[/cyan]",
        ]
        if final_video_path:
            lines.append(f"Final video: [cyan]{final_video_path}[/cyan]")
        self.reporter.panel("\n".join(lines), title="Full-Auto Complete")

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
