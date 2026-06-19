from __future__ import annotations

from pathlib import Path

from feverslop.adapters.comfyui_client import ComfyUIClient


class ComfyUIVideoAssetUploader:
    def __init__(self, client: ComfyUIClient):
        self.client = client

    def resolve_audio_name(
        self,
        audio_file: str | Path,
        *,
        upload_audio: bool,
        uploaded_audio_name: str | None,
    ) -> str:
        audio_file = Path(audio_file)
        if not upload_audio:
            return uploaded_audio_name or audio_file.name

        audio_upload = self.client.upload_file_via_image_endpoint(
            audio_file,
            subfolder="feverslop/audio",
            file_type="input",
            overwrite=True,
        )
        return self.comfy_path_from_upload(audio_upload)

    def resolve_startframe_name(
        self,
        startframe_path: str | Path,
        *,
        upload_startframes: bool,
    ) -> str:
        startframe_path = Path(startframe_path)
        if not upload_startframes:
            return startframe_path.name

        image_upload = self.client.upload_image(
            startframe_path,
            subfolder="feverslop/storyboard",
            file_type="input",
            overwrite=True,
        )
        return self.comfy_path_from_upload(image_upload)

    @staticmethod
    def comfy_path_from_upload(upload_response: dict) -> str:
        name = upload_response.get("name")
        subfolder = upload_response.get("subfolder", "")
        if not name:
            raise ValueError(f"Unexpected ComfyUI upload response: {upload_response}")
        return f"{subfolder}/{name}" if subfolder else name
