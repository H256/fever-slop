from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from feverslop.domain.render_plan import RenderPlan
from feverslop.ports.artifacts import ArtifactStore
from feverslop.ports.reporting import ConsoleReporter, Reporter
from feverslop.ports.rendering import VideoRenderBackend, VideoRenderRequest, WorkflowAnchorConfig


@dataclass(frozen=True)
class RenderVideoScenesRequest:
    render_plan_path: Path
    workflow_path: Path
    audio_file: Path
    storyboard_dir: Path
    output_dir: Path
    render_mode: str = "single_prompt"
    single_prompt_workflow_path: Path | None = None
    limit: int | None = None
    scene_numbers: set[int] | None = None
    skip_existing: bool = True
    uploaded_audio_name: str | None = None
    upload_audio: bool = True
    upload_startframes: bool = True
    anchors: WorkflowAnchorConfig = WorkflowAnchorConfig()
    on_scene_complete: Callable[[Path, int, int], None] | None = None


class RenderVideoScenesUseCase:
    def __init__(
        self,
        backend: VideoRenderBackend,
        artifact_store: ArtifactStore,
        console: object | None = None,
        reporter: Reporter | None = None,
    ):
        self.backend = backend
        self.artifact_store = artifact_store
        self.reporter = reporter or (ConsoleReporter(console) if console is not None else None)

    def execute(self, request: RenderVideoScenesRequest) -> list[Path]:
        plan = RenderPlan.from_dicts(
            self.artifact_store.read_render_plan(request.render_plan_path)
        ).select(
            scene_numbers=request.scene_numbers,
            limit=request.limit,
        )

        rendered: list[Path] = []
        total = len(plan.scenes)
        for scene in plan.scenes:
            final_path = request.output_dir / "final" / f"scene_{scene.scene_number:04}.mp4"
            if request.skip_existing and final_path.exists():
                rendered.append(final_path)
                self._log_scene_available(final_path, len(rendered), total, skipped=True)
                if request.on_scene_complete:
                    request.on_scene_complete(final_path, len(rendered), total)
                continue

            output_path = self.backend.render_video(
                VideoRenderRequest(
                    scene=scene.to_dict(),
                    scene_number=scene.scene_number,
                    prompt=scene.video_prompt,
                    workflow_path=request.workflow_path,
                    output_dir=request.output_dir,
                    audio_file=request.audio_file,
                    storyboard_dir=request.storyboard_dir,
                    render_mode=request.render_mode,
                    single_prompt_workflow_path=request.single_prompt_workflow_path,
                    skip_existing=request.skip_existing,
                    uploaded_audio_name=request.uploaded_audio_name,
                    upload_audio=request.upload_audio,
                    upload_startframes=request.upload_startframes,
                    anchors=request.anchors,
                )
            )
            rendered.append(output_path)
            self._log_scene_available(output_path, len(rendered), total, skipped=False)
            if request.on_scene_complete:
                request.on_scene_complete(output_path, len(rendered), total)

        return rendered

    def _log_scene_available(self, output_path: Path, completed: int, total: int, *, skipped: bool) -> None:
        if self.reporter is None:
            return
        verb = "Available" if skipped else "Rendered"
        self.reporter.message(
            f"[green]OK[/green] {verb} scene {completed}/{total}: [cyan]{output_path}[/cyan]"
        )
