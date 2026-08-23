from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from feverslop.adapters.reporting import ConsoleReporter
from feverslop.domain.render_plan import RenderPlan
from feverslop.ports.artifacts import ArtifactStore
from feverslop.ports.rendering import (
    VideoRenderBackend,
    VideoRenderRequest,
    WorkflowAnchorConfig,
)
from feverslop.ports.reporting import Reporter
from feverslop.utils.io import file_is_valid


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
        console: Any | None = None,
        reporter: Reporter | None = None,
    ):
        self.backend = backend
        self.artifact_store = artifact_store
        self.reporter = reporter or (ConsoleReporter(console) if console is not None else None)

    def execute(self, request: RenderVideoScenesRequest) -> list[Path]:
        render_plan_data = self.artifact_store.read_render_plan(request.render_plan_path)
        plan = RenderPlan.from_dicts(
            render_plan_data,
        ).select(
            scene_numbers=request.scene_numbers,
            limit=request.limit,
        )

        rendered: list[Path] = []
        total = len(plan.scenes)
        for scene in plan.scenes:
            scene_payload = scene.to_dict()
            video_request = VideoRenderRequest(
                scene=scene_payload,
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
                render_plan_path=request.render_plan_path,
            )
            final_path = request.output_dir / "final" / f"scene_{scene.scene_number:04}.mp4"
            direct_path = request.output_dir / f"scene_{scene.scene_number:04}.mp4"
            per_scene_path = request.output_dir / f"scene_{scene.scene_number:04}" / "final.mp4"
            existing_path = (
                final_path if file_is_valid(final_path)
                else (direct_path if file_is_valid(direct_path)
                      else (per_scene_path if file_is_valid(per_scene_path) else None))
            )
            if request.skip_existing and existing_path:
                ensure_manifest = getattr(self.backend, "ensure_scene_manifest", None)
                if ensure_manifest is not None:
                    ensure_manifest(video_request)
                rendered.append(existing_path)
                self._log_scene_available(existing_path, len(rendered), total, skipped=True)
                if request.on_scene_complete:
                    request.on_scene_complete(existing_path, len(rendered), total)
                continue

            randomize_seed = bool(getattr(self.backend, "randomize_seed", False))
            if randomize_seed:
                scene_payload["seed"] = random.SystemRandom().randint(0, 2**63 - 1)
                render_plan_data = [
                    (
                        scene_payload
                        if int(item["scene"]) == scene.scene_number
                        else item
                    )
                    for item in render_plan_data
                ]
                self.artifact_store.write_render_plan(request.render_plan_path, render_plan_data)
                original_randomize_seed = self.backend.randomize_seed
                self.backend.randomize_seed = False
                try:
                    output_path = self.backend.render_video(video_request)
                finally:
                    self.backend.randomize_seed = original_randomize_seed
            else:
                output_path = self.backend.render_video(video_request)
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
            f"[green]OK[/green] {verb} scene {completed}/{total}: [cyan]{output_path}[/cyan]",
        )
