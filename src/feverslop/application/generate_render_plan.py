from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from feverslop.config.app_config import AppConfig
from feverslop.application.pipeline_context import GenerateRenderPlanContext
from feverslop.config.project_config import ProjectConfig, ProjectPaths
from feverslop.ports.artifacts import ArtifactStore


@dataclass(frozen=True)
class GenerateRenderPlanRequest:
    project_config_path: Path
    app_config_path: Path = Path("app_config.json")
    concept_batch_size: int = 0
    render_storyboard: bool = False
    zimage_workflow_path: Path | None = None


@dataclass(frozen=True)
class GenerateRenderPlanResult:
    render_plan_path: Path
    scene_count: int
    total_frames: int
    total_duration_seconds: float


class GenerateRenderPlanUseCase:
    def __init__(
        self,
        console: Console | None = None,
        pipeline_services: list[Any] | None = None,
        artifact_store: ArtifactStore | None = None,
        storyboard_renderer_factory: Callable[[AppConfig, Path, Path], Any] | None = None,
    ):
        self.console = console or Console()
        self.pipeline_services = pipeline_services if pipeline_services is not None else []
        self.artifact_store = artifact_store
        self.storyboard_renderer_factory = storyboard_renderer_factory

    def build_default_pipeline_services(self) -> list[Any]:
        return list(self.pipeline_services)

    def execute_services(self, context: GenerateRenderPlanContext | dict[str, Any]) -> GenerateRenderPlanContext | dict[str, Any]:
        for service in self.pipeline_services:
            context = service.execute(context)
        return context

    def log_step(self, title: str):
        self.console.print()
        self.console.rule(f"[bold cyan]{title}[/bold cyan]")

    def log_file(self, label: str, path: Path):
        self.console.print(f"[green]âœ“[/green] {label}: [cyan]{path}[/cyan]")

    def run_spinner(self, description: str, func: Callable[[], Any]):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            progress.add_task(description, total=None)
            return func()

    def execute(self, request: GenerateRenderPlanRequest) -> GenerateRenderPlanResult:
        started_at = time.time()
        if self.artifact_store is None:
            raise ValueError("GenerateRenderPlanUseCase requires an artifact_store")

        config = ProjectConfig.load(request.project_config_path)
        paths = ProjectPaths.from_config(config)
        app_config = AppConfig.load(request.app_config_path)
        video_settings = config.to_video_settings()
        song_id = getattr(config, "song_id", None) or getattr(config, "project_name", "") or config.input_audio.stem

        paths.ensure_output_dirs()
        timeline_dir = paths.timeline_dir
        prompts_dir = paths.prompts_dir
        render_dir = paths.render_dir

        context = GenerateRenderPlanContext(
            request=request,
            config=config,
            paths=paths,
            app_config=app_config,
            video_settings=video_settings,
            song_id=song_id,
            artifact_store=self.artifact_store,
            console=self.console,
            log_step=self.log_step,
            log_file=self.log_file,
            run_spinner=self.run_spinner,
            timeline_json=timeline_dir / f"timeline_{song_id}.json",
            beat_json=timeline_dir / f"beat_data_{song_id}.json",
            scene_srt_raw=timeline_dir / f"scenes_{song_id}_raw.srt",
            scene_srt=timeline_dir / f"scenes_{song_id}.srt",
            stage1_segments_json=timeline_dir / f"stage1_segments_{song_id}.json",
            ltx_prompt_relay_json=prompts_dir / f"ltx_prompt_relay_{song_id}.json",
            resolved_context_json=prompts_dir / f"resolved_context_{song_id}.json",
            concept_prompts_json=prompts_dir / f"concept_prompts_{song_id}.json",
            scene_details_json=prompts_dir / f"scene_details_{song_id}.json",
            scene_prompts_json=prompts_dir / f"scene_prompts_{song_id}.json",
            render_plan_json=render_dir / f"render_plan_{song_id}.json",
        )

        self.console.print(Panel.fit(
            f"[bold]Music Video Pipeline[/bold]\n\n"
            f"Project: [cyan]{config.project_name}[/cyan]\n"
            f"Input: [cyan]{config.input_audio}[/cyan]\n"
            f"Output: [cyan]{config.output_dir}[/cyan]\n"
            f"FPS: [yellow]{video_settings.fps}[/yellow]\n"
            f"Resolution: [yellow]{video_settings.width}x{video_settings.height}[/yellow]\n"
            f"LLM: [yellow]{app_config.llm.model}[/yellow] @ [cyan]{app_config.llm.base_url}[/cyan]",
            title="Startup",
            border_style="cyan",
        ))

        if not config.input_audio.exists():
            raise FileNotFoundError(config.input_audio)

        context = self.execute_services(context)
        render_plan = context["render_plan"]
        total_frames = sum(scene["frame_count"] for scene in render_plan)
        total_duration = sum(scene["duration_seconds"] for scene in render_plan)
        self.print_summary(
            render_plan=render_plan,
            total_frames=total_frames,
            total_duration=total_duration,
            video_settings=video_settings,
            render_plan_json=context["render_plan_json"],
            elapsed=time.time() - started_at,
        )

        if request.render_storyboard:
            self.render_storyboard(
                request=request,
                app_config=app_config,
                render_dir=render_dir,
                render_plan_json=context["render_plan_json"],
            )

        return GenerateRenderPlanResult(
            render_plan_path=context["render_plan_json"],
            scene_count=len(render_plan),
            total_frames=total_frames,
            total_duration_seconds=total_duration,
        )

    def print_summary(
        self,
        *,
        render_plan: list[dict],
        total_frames: int,
        total_duration: float,
        video_settings: Any,
        render_plan_json: Path,
        elapsed: float,
    ) -> None:
        summary = Table(title="Render Plan Summary")
        summary.add_column("Metric", style="bold")
        summary.add_column("Value", style="yellow")
        summary.add_row("Scenes / Cuts", str(len(render_plan)))
        summary.add_row("Total Frames", str(total_frames))
        summary.add_row("Total Duration", f"{total_duration:.2f}s")
        summary.add_row("FPS", str(video_settings.fps))
        summary.add_row("Resolution", f"{video_settings.width}x{video_settings.height}")
        self.console.print(summary)
        self.console.print(Panel.fit(
            f"[bold green]Done.[/bold green]\n\n"
            f"Elapsed: [yellow]{elapsed:.1f}s[/yellow]\n"
            f"Render plan: [cyan]{render_plan_json}[/cyan]",
            title="Pipeline Complete",
            border_style="green",
        ))

    def render_storyboard(
        self,
        *,
        request: GenerateRenderPlanRequest,
        app_config: AppConfig,
        render_dir: Path,
        render_plan_json: Path,
    ) -> None:
        if not request.zimage_workflow_path:
            raise ValueError("--zimage-workflow is required when --render-storyboard is used")
        if self.storyboard_renderer_factory is None:
            raise ValueError("Storyboard rendering requires a storyboard_renderer_factory")

        renderer = self.storyboard_renderer_factory(app_config, render_dir, request.zimage_workflow_path)
        rendered = renderer.render_storyboard(render_plan_path=render_plan_json)
        self.console.print(
            f"[green]âœ“[/green] Rendered storyboard frames: [yellow]{len(rendered)}[/yellow]"
        )
