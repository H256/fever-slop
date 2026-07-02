from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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
    width: int = 1280
    height: int = 704
    fps: int = 24
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
        console: Console | None = None,
    ):
        self.brief_generator = brief_generator
        self.song_generator = song_generator
        self.project_scaffold = project_scaffold
        self.pipeline_runner = pipeline_runner
        self.console = console or Console()

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
        )
        self.log_file("Project config", scaffold.project_config_path)
        self.log_file("Lyrics", scaffold.lyrics_path)
        self.log_file("Song spec", scaffold.song_spec_path)

        final_video_path = None
        if request.run_video_pipeline:
            if self.pipeline_runner is None:
                raise ValueError("FullAutoUseCase requires a pipeline_runner when run_video_pipeline is true")
            self.log_step("4. Running video pipeline")
            final_video_path = self.pipeline_runner.run(
                project_config_path=scaffold.project_config_path,
                options=dict(request.runner_options),
            )
            if final_video_path:
                self.log_file("Final video", final_video_path)
        else:
            self.console.print("[yellow]Skipping video pipeline; project is prepared for later rendering.[/yellow]")

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
        self.console.print()
        self.console.rule(f"[bold cyan]{title}[/bold cyan]")

    def log_file(self, label: str, path: Path) -> None:
        self.console.print(f"[green]OK[/green] {label}: [cyan]{path}[/cyan]")

    def _print_startup(self, *, request: FullAutoRequest, project_slug: str) -> None:
        self.console.print(
            Panel.fit(
                f"[bold]Full-Auto ACE-Step Pipeline[/bold]\n\n"
                f"Project: [cyan]{project_slug}[/cyan]\n"
                f"Duration: [yellow]{float(request.duration_seconds):.1f}s[/yellow]\n"
                f"Resolution: [yellow]{int(request.width)}x{int(request.height)} @ {int(request.fps)}fps[/yellow]\n"
                f"Language: [yellow]{request.language}[/yellow]\n"
                f"Seed: [yellow]{int(request.seed)}[/yellow]\n"
                f"Video pipeline: [yellow]{'on' if request.run_video_pipeline else 'off'}[/yellow]",
                title="Startup",
                border_style="cyan",
            )
        )

    def _print_song_spec(self, spec: SongSpec) -> None:
        table = Table(title="Generated Song Brief")
        table.add_column("Field", style="bold")
        table.add_column("Value", style="yellow")
        table.add_row("Title", spec.title)
        table.add_row("BPM", str(spec.bpm))
        table.add_row("Duration", f"{float(spec.duration_seconds):.1f}s")
        table.add_row("Language", spec.language)
        table.add_row("Key", spec.keyscale)
        self.console.print(table)
        self.console.print(Panel(spec.tags, title="ACE-Step Tags", border_style="green"))
        self.console.print(Panel(spec.visual_story_idea, title="Video Story", border_style="green"))

    def _print_complete(self, *, scaffold, final_video_path: Path | None) -> None:
        lines = [
            "[bold green]Done.[/bold green]",
            "",
            f"Project config: [cyan]{scaffold.project_config_path}[/cyan]",
            f"Audio: [cyan]{scaffold.audio_path}[/cyan]",
        ]
        if final_video_path:
            lines.append(f"Final video: [cyan]{final_video_path}[/cyan]")
        self.console.print(
            Panel.fit(
                "\n".join(lines),
                title="Full-Auto Complete",
                border_style="green",
            )
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
