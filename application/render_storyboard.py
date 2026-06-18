from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from adapters.local_artifacts import JsonArtifactStore
from domain.render_plan import RenderPlan
from ports.artifacts import ArtifactStore
from ports.rendering import ImageRenderBackend, ImageRenderRequest, WorkflowAnchorConfig


@dataclass(frozen=True)
class RenderStoryboardRequest:
    render_plan_path: Path
    workflow_path: Path
    output_dir: Path
    limit: int | None = None
    scene_numbers: set[int] | None = None
    skip_existing: bool = True
    negative_prompt: str = ""
    character_lora_strength: float = 1.0
    anchors: WorkflowAnchorConfig = WorkflowAnchorConfig()


class RenderStoryboardUseCase:
    def __init__(
        self,
        backend: ImageRenderBackend,
        artifact_store: ArtifactStore | None = None,
    ):
        self.backend = backend
        self.artifact_store = artifact_store or JsonArtifactStore()

    def execute(self, request: RenderStoryboardRequest) -> list[Path]:
        plan = RenderPlan.from_dicts(
            self.artifact_store.read_render_plan(request.render_plan_path)
        ).select(
            scene_numbers=request.scene_numbers,
            limit=request.limit,
        )

        rendered: list[Path] = []
        for scene in plan.scenes:
            output_path = request.output_dir / f"scene_{scene.scene_number:04}.png"
            if request.skip_existing and output_path.exists():
                rendered.append(output_path)
                continue

            rendered.append(
                self.backend.render_image(
                    ImageRenderRequest(
                        scene=scene.to_dict(),
                        scene_number=scene.scene_number,
                        prompt=scene.z_image_prompt,
                        workflow_path=request.workflow_path,
                        output_dir=request.output_dir,
                        skip_existing=request.skip_existing,
                        negative_prompt=request.negative_prompt,
                        character_lora_strength=request.character_lora_strength,
                        anchors=request.anchors,
                    )
                )
            )

        return rendered
