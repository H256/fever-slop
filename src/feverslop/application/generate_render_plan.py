from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable

from feverslop.application.pipeline_context import GenerateRenderPlanContext
from feverslop.domain.scene_duration_limits import ResolvedSceneDurationPolicy
from feverslop.errors import FeverSlopConfigError, FeverSlopValidationError
from feverslop.ports.artifacts import ArtifactStore
from feverslop.ports.reporting import Reporter
from feverslop.adapters.reporting import ConsoleReporter, NullReporter


@dataclass(frozen=True)
class GenerateRenderPlanRequest:
    project_config_path: Path
    app_config_path: Path = Path("app_config.json")
    concept_batch_size: int = 0
    render_storyboard: bool = False
    zimage_workflow_path: Path | None = None
    video_workflow_paths: tuple[Path, ...] = ()
    rolling_frame_profile: str = "original"
    defer_h3_until_references: bool = False


@dataclass(frozen=True)
class GenerateRenderPlanExecutionRequest:
    source_request: GenerateRenderPlanRequest | Any
    config: Any
    paths: Any
    app_config: Any
    video_settings: Any
    song_id: str
    scene_duration_policy: ResolvedSceneDurationPolicy | None = None


@dataclass(frozen=True)
class GenerateRenderPlanResult:
    render_plan_path: Path
    scene_count: int
    total_frames: int
    total_duration_seconds: float


class GenerateRenderPlanUseCase:
    def __init__(
        self,
        console: Any | None = None,
        reporter: Reporter | None = None,
        pipeline_services: list[Any] | None = None,
        artifact_store: ArtifactStore | None = None,
        storyboard_renderer_factory: Callable[[Any, Path, Path], Any] | None = None,
    ):
        if reporter is not None:
            self.reporter = reporter
        elif console is not None:
            self.reporter = ConsoleReporter(console)
        else:
            self.reporter = NullReporter()
        self.pipeline_services = pipeline_services if pipeline_services is not None else []
        self.artifact_store = artifact_store
        self.storyboard_renderer_factory = storyboard_renderer_factory

    def build_default_pipeline_services(self) -> list[Any]:
        return list(self.pipeline_services)

    def execute_services(self, context: GenerateRenderPlanContext | dict[str, Any]) -> GenerateRenderPlanContext | dict[str, Any]:
        request = context["request"]
        defer_h3 = getattr(request, "defer_h3_until_references", False)
        for service in self.pipeline_services:
            if defer_h3 and getattr(service, "defer_until_references", False):
                continue
            context = service.execute(context)
        return context

    def log_step(self, title: str):
        self.reporter.step(title)

    def log_file(self, label: str, path: Path):
        self.reporter.file(label, path)

    def run_spinner(self, description: str, func: Callable[[], Any]):
        return self.reporter.run_progress(description, func)

    def execute(self, request: GenerateRenderPlanExecutionRequest) -> GenerateRenderPlanResult:
        started_at = time.time()
        if self.artifact_store is None:
            raise FeverSlopConfigError("GenerateRenderPlanUseCase requires an artifact_store")

        config = request.config
        paths = request.paths
        app_config = request.app_config
        video_settings = request.video_settings
        song_id = request.song_id

        paths.ensure_output_dirs()
        timeline_dir = paths.timeline_dir
        prompts_dir = paths.prompts_dir
        render_dir = paths.render_dir

        context = GenerateRenderPlanContext(
            request=request.source_request,
            config=config,
            paths=paths,
            app_config=app_config,
            video_settings=video_settings,
            song_id=song_id,
            scene_duration_policy=request.scene_duration_policy,
            artifact_store=self.artifact_store,
            reporter=self.reporter,
            console=_ReporterConsole(self.reporter),
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
            h3_prompts_json=prompts_dir / f"h3_prompts_{song_id}.json",
            render_plan_json=paths.artifact_layout.base_plan,
        )

        self.reporter.panel(
            f"Music Video Pipeline\n\n"
            f"Project: [cyan]{config.project_name}[/cyan]\n"
            f"Input: [cyan]{config.input_audio}[/cyan]\n"
            f"Output: [cyan]{config.output_dir}[/cyan]\n"
            f"FPS: [yellow]{video_settings.fps}[/yellow]\n"
            f"Resolution: [yellow]{video_settings.width}x{video_settings.height}[/yellow]\n"
            f"LLM: [yellow]{app_config.llm.model}[/yellow] @ [cyan]{app_config.llm.base_url}[/cyan]",
            title="Startup",
        )

        if not config.input_audio.exists():
            raise FileNotFoundError(config.input_audio)

        self.report_scene_duration_clamp(request.scene_duration_policy)
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

        if getattr(request.source_request, "render_storyboard", False):
            self.render_storyboard(
                request=request.source_request,
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

    def report_scene_duration_clamp(
        self,
        policy: ResolvedSceneDurationPolicy | None,
    ) -> None:
        if policy is None or not policy.clamped:
            return
        limiting_workflow = policy.limiting_workflow or "Default ComfyUI video limit"
        self.reporter.panel(
            f"Requested scene duration: {policy.requested_min_seconds:.3f}s.."
            f"{policy.requested_max_seconds:.3f}s\n"
            f"Effective scene duration: {policy.effective_min_seconds:.3f}s.."
            f"{policy.effective_max_seconds:.3f}s\n"
            f"Render limit: {policy.max_render_duration_seconds:.3f}s including "
            f"{policy.preroll_frames} pre-roll and {policy.tail_frames} tail frames\n"
            f"Limiting workflow: {limiting_workflow}",
            title="Scene duration limit",
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
        self.reporter.table(
            "Render Plan Summary",
            ["Metric", "Value"],
            [
                ["Scenes / Cuts", str(len(render_plan))],
                ["Total Frames", str(total_frames)],
                ["Total Duration", f"{total_duration:.2f}s"],
                ["FPS", str(video_settings.fps)],
                ["Resolution", f"{video_settings.width}x{video_settings.height}"],
            ],
        )
        self.reporter.panel(
            f"[bold green]Done.[/bold green]\n\n"
            f"Elapsed: [yellow]{elapsed:.1f}s[/yellow]\n"
            f"Render plan: [cyan]{render_plan_json}[/cyan]",
            title="Pipeline Complete",
        )

    def render_storyboard(
        self,
        *,
        request: Any,
        app_config: Any,
        render_dir: Path,
        render_plan_json: Path,
    ) -> None:
        if not request.zimage_workflow_path:
            raise FeverSlopValidationError("--zimage-workflow is required when --render-storyboard is used")
        if self.storyboard_renderer_factory is None:
            raise FeverSlopConfigError("Storyboard rendering requires a storyboard_renderer_factory")

        renderer = self.storyboard_renderer_factory(app_config, render_dir, request.zimage_workflow_path)
        rendered = renderer.render_storyboard(render_plan_path=render_plan_json)
        self.reporter.message(
            f"[green]OK[/green] Rendered storyboard frames: [yellow]{len(rendered)}[/yellow]"
        )


class _ReporterConsole:
    def __init__(self, reporter: Reporter):
        self.reporter = reporter

    def print(self, *values: object, **_kwargs: object) -> None:
        self.reporter.message(" ".join(str(value) for value in values))

    def rule(self, title: str) -> None:
        self.reporter.step(title)
