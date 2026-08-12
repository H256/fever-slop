from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from feverslop.domain.render_plan import RenderPlan
from feverslop.ports.artifacts import ArtifactStore
from feverslop.ports.llm import StoryboardPromptTransformerPort
from feverslop.ports.rendering import ImageRenderBackend, ImageRenderRequest, WorkflowAnchorConfig
from feverslop.utils.io import file_is_valid


@dataclass(frozen=True)
class RenderStoryboardRequest:
    render_plan_path: Path
    workflow_path: Path
    output_dir: Path
    limit: int | None = None
    scene_numbers: set[int] | None = None
    skip_existing: bool = True
    negative_prompt: str = ""
    character_lora_strength: float | None = None
    anchors: WorkflowAnchorConfig = WorkflowAnchorConfig()
    on_frame_complete: Callable[[Path, int, int], None] | None = None


class RenderStoryboardUseCase:
    def __init__(
        self,
        backend: ImageRenderBackend,
        artifact_store: ArtifactStore,
        prompt_transformer: StoryboardPromptTransformerPort | None = None,
        positive_prompt_input: str | None = None,
    ):
        self.backend = backend
        self.artifact_store = artifact_store
        self.prompt_transformer = prompt_transformer
        self.positive_prompt_input = positive_prompt_input

    def execute(self, request: RenderStoryboardRequest) -> list[Path]:
        plan = RenderPlan.from_dicts(
            self.artifact_store.read_render_plan(request.render_plan_path)
        ).select(
            scene_numbers=request.scene_numbers,
            limit=request.limit,
        )

        rendered: list[Path] = []
        total = len(plan.scenes)
        for scene in plan.scenes:
            output_path = request.output_dir / f"scene_{scene.scene_number:04}.png"
            if request.skip_existing and file_is_valid(output_path):
                rendered.append(output_path)
                if request.on_frame_complete:
                    request.on_frame_complete(output_path, len(rendered), total)
                continue

            prompt = scene.z_image_prompt
            if self.prompt_transformer is not None:
                prompt = self.prompt_transformer.transform_prompt(
                    scene_number=scene.scene_number,
                    original_prompt=scene.z_image_prompt,
                    width=scene.width,
                    height=scene.height,
                )

            anchors = request.anchors
            if self.positive_prompt_input is not None:
                anchors = replace(anchors, positive_prompt_input=self.positive_prompt_input)

            rendered_path = self.backend.render_image(
                ImageRenderRequest(
                    scene=scene.to_dict(),
                    scene_number=scene.scene_number,
                    prompt=prompt,
                    workflow_path=request.workflow_path,
                    output_dir=request.output_dir,
                    width=scene.width,
                    height=scene.height,
                    skip_existing=request.skip_existing,
                    negative_prompt=request.negative_prompt,
                    character_lora_strength=request.character_lora_strength,
                    anchors=anchors,
                )
            )
            rendered.append(rendered_path)
            if request.on_frame_complete:
                request.on_frame_complete(rendered_path, len(rendered), total)

        return rendered
