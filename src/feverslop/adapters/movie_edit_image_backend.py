from __future__ import annotations

import json
from pathlib import Path

from feverslop.adapters.comfyui_model_resolver import NoOpComfyUIModelResolver
from feverslop.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader
from feverslop.adapters.workflow_patcher import WorkflowPatcher
from feverslop.errors import FeverSlopRenderError


class MovieTwoRefEditImageBackend:
    def __init__(self, *, client: object, workflow_path: str | Path, model_resolver=None):
        self.client = client
        self.workflow_path = Path(workflow_path)
        self.model_resolver = model_resolver or NoOpComfyUIModelResolver()

    def render_edit(
        self,
        *,
        scene_number: int,
        prompt: str,
        plate_image: str | Path,
        character_image: str | Path,
        output_dir: str | Path,
        pass_number: int,
        negative_prompt: str = "",
    ) -> Path:
        workflow = json.loads(self.workflow_path.read_text(encoding="utf-8-sig"))
        patcher = WorkflowPatcher(workflow)

        plate_name = self._upload_image(plate_image)
        character_name = self._upload_image(character_image)
        patcher.set_input_by_title("#BASE_IMAGE", "image", plate_name)
        patcher.set_input_by_title("#CHARACTER_REF", "image", character_name)
        patcher.set_existing_input_by_title_any("#PROMPT_POSITIVE", "text", prompt)
        patcher.set_existing_input_by_title_any("#PROMPT_NEGATIVE", "text", negative_prompt)
        prefix = f"movie_edit/scene_{int(scene_number):04}_pass_{int(pass_number):02}"
        patcher.set_input_by_title("#SAVE_IMAGE", "filename_prefix", prefix)

        resolved = self.model_resolver.resolve_workflow_models(
            patcher.get(),
            workflow_path=self.workflow_path,
        )
        prompt_id = self.client.queue_prompt(resolved)
        history = self.client.wait_for_completion(prompt_id)
        images = self.client.extract_output_images(history)
        if not images:
            raise FeverSlopRenderError(f"No edit image output for scene {scene_number} pass {pass_number}")

        first = images[0]
        return self.client.download_view_file(
            filename=first["filename"],
            subfolder=first.get("subfolder", ""),
            file_type=first.get("type", "output"),
            output_path=Path(output_dir) / f"scene_{int(scene_number):04}_pass_{int(pass_number):02}.png",
        )

    def _upload_image(self, image_path: str | Path) -> str:
        upload = self.client.upload_image(
            Path(image_path),
            subfolder="feverslop/movie_edit",
            file_type="input",
            overwrite=True,
        )
        return ComfyUIVideoAssetUploader.comfy_path_from_upload(upload)
