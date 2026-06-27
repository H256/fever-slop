from __future__ import annotations

from pathlib import Path
import json

from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_model_resolver import NoOpComfyUIModelResolver
from feverslop.adapters.comfyui_video_backend import ComfyUIVideoRenderBackend
from feverslop.ports.rendering import ImageRenderRequest
from feverslop.adapters.workflow_patcher import WorkflowPatcher

__all__ = ["ComfyUIImageBackend", "ComfyUIVideoRenderBackend"]


class ComfyUIImageBackend:
    def __init__(
        self,
        client: ComfyUIClient,
        workflow_path: str | Path,
        output_dir: str | Path,
        seed_node_title: str | None = None,
        seed_input_name: str = "seed",
        filename_prefix_input_name: str = "filename_prefix",
        model_resolver=None,
    ):
        self.client = client
        self.workflow_path = Path(workflow_path)
        self.output_dir = Path(output_dir)
        self.seed_node_title = seed_node_title
        self.seed_input_name = seed_input_name
        self.filename_prefix_input_name = filename_prefix_input_name
        self.model_resolver = model_resolver or NoOpComfyUIModelResolver()

    def load_workflow(self) -> dict:
        return json.loads(self.workflow_path.read_text(encoding="utf-8-sig"))

    def render_image(self, request: ImageRenderRequest) -> Path:
        workflow = self.load_workflow()
        patcher = WorkflowPatcher(workflow)

        scene_number = int(request.scene_number)
        anchors = request.anchors

        patcher.set_existing_input_by_title_any(
            anchors.positive_prompt_title,
            anchors.positive_prompt_input,
            request.prompt,
        )

        if anchors.negative_prompt_title:
            patcher.set_input_by_title(
                anchors.negative_prompt_title,
                "text",
                request.negative_prompt,
            )

        if anchors.character_lora_title and request.character_lora_strength is not None:
            patcher.patch_lora_strength_by_title(
                anchors.character_lora_title,
                request.character_lora_strength,
            )

        if anchors.reference_image_title and request.reference_image is not None:
            image_upload = self.client.upload_image(
                request.reference_image,
                subfolder="feverslop/references",
                file_type="input",
                overwrite=True,
            )
            patcher.set_input_by_title(
                anchors.reference_image_title,
                anchors.reference_image_input,
                self.client.comfy_path_from_upload(image_upload)
                if hasattr(self.client, "comfy_path_from_upload")
                else _comfy_path_from_upload(image_upload),
            )

        if self.seed_node_title:
            patcher.set_input_by_title(
                self.seed_node_title,
                self.seed_input_name,
                scene_number,
            )

        if anchors.save_image_title:
            patcher.set_input_by_title(
                anchors.save_image_title,
                self.filename_prefix_input_name,
                f"storyboard/scene_{scene_number:04}",
            )

        if request.width is not None and anchors.width_title:
            width_patched = patcher.try_set_existing_input_by_title(
                anchors.width_title,
                anchors.width_input,
                int(request.width),
            )
        else:
            width_patched = False

        if request.height is not None and anchors.height_title:
            height_patched = patcher.try_set_existing_input_by_title(
                anchors.height_title,
                anchors.height_input,
                int(request.height),
            )
        else:
            height_patched = False

        if request.width is not None and request.height is not None and not (width_patched and height_patched):
            self._try_patch_dimensions_node(patcher, int(request.width), int(request.height))

        workflow = self.model_resolver.resolve_workflow_models(
            patcher.get(),
            workflow_path=self.workflow_path,
        )
        prompt_id = self.client.queue_prompt(workflow)
        history = self.client.wait_for_completion(prompt_id)
        images = self.client.extract_output_images(history)

        if not images:
            raise RuntimeError(f"No image output for scene {scene_number}")

        first = images[0]
        return self.client.download_view_file(
            filename=first["filename"],
            subfolder=first.get("subfolder", ""),
            file_type=first.get("type", "output"),
            output_path=request.output_dir / f"scene_{scene_number:04}.png",
        )

    @staticmethod
    def _try_patch_dimensions_node(patcher: WorkflowPatcher, width: int, height: int) -> None:
        try:
            patcher.set_existing_input_by_title("#DIMENSIONS", "width", width)
            patcher.set_existing_input_by_title("#DIMENSIONS", "height", height)
        except KeyError:
            return


def _comfy_path_from_upload(upload_response: dict) -> str:
    name = upload_response.get("name")
    subfolder = upload_response.get("subfolder", "")
    if not name:
        raise ValueError(f"Unexpected ComfyUI upload response: {upload_response}")
    return f"{subfolder}/{name}" if subfolder else name
