from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from autoprompter.adapters.local_artifacts import JsonArtifactStore
from autoprompter.domain.render_plan import RenderPlan
from autoprompter.ports.artifacts import ArtifactStore
from autoprompter.ports.rendering import VideoRenderBackend, VideoRenderRequest, WorkflowAnchorConfig


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


class RenderVideoScenesUseCase:
    def __init__(
        self,
        backend: VideoRenderBackend,
        artifact_store: ArtifactStore | None = None,
    ):
        self.backend = backend
        self.artifact_store = artifact_store or JsonArtifactStore()

    def execute(self, request: RenderVideoScenesRequest) -> list[Path]:
        plan = RenderPlan.from_dicts(
            self.artifact_store.read_render_plan(request.render_plan_path)
        ).select(
            scene_numbers=request.scene_numbers,
            limit=request.limit,
        )

        rendered: list[Path] = []
        for scene in plan.scenes:
            final_path = request.output_dir / "final" / f"scene_{scene.scene_number:04}.mp4"
            if request.skip_existing and final_path.exists():
                rendered.append(final_path)
                continue

            rendered.append(
                self.backend.render_video(
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
            )

        return rendered
