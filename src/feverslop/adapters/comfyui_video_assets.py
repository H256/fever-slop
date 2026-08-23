from __future__ import annotations

import hashlib
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
            upload_name=ComfyUIVideoAssetUploader.content_addressed_name(audio_file),
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

    def resolve_reference_image_name(
        self,
        image_path: str | Path,
        *,
        upload_references: bool = True,
    ) -> str:
        image_path = Path(image_path)
        if not upload_references:
            return image_path.name

        image_upload = self.client.upload_image(
            image_path,
            subfolder="feverslop/references",
            file_type="input",
            overwrite=True,
            upload_name=ComfyUIVideoAssetUploader.content_addressed_name(image_path),
        )
        return self.comfy_path_from_upload(image_upload)


    def resolve_reference_video_name(
        self,
        video_path: str | Path,
        *,
        upload_references: bool = True,
    ) -> str:
        video_path = Path(video_path)
        if not upload_references:
            return video_path.name
        image_upload = self.client.upload_image(
            video_path,
            subfolder="feverslop/references",
            file_type="input",
            overwrite=True,
            upload_name=ComfyUIVideoAssetUploader.content_addressed_name(video_path),
        )
        return self.comfy_path_from_upload(image_upload)

    def resolve_reference_audio_name(
        self,
        audio_path: str | Path,
        *,
        upload_references: bool = True,
    ) -> str:
        audio_path = Path(audio_path)
        if not upload_references:
            return audio_path.name
        image_upload = self.client.upload_image(
            audio_path,
            subfolder="feverslop/references",
            file_type="input",
            overwrite=True,
            upload_name=ComfyUIVideoAssetUploader.content_addressed_name(audio_path),
        )
        return self.comfy_path_from_upload(image_upload)

    @staticmethod
    def content_addressed_name(file_path: Path) -> str:
        if not file_path.exists():
            return file_path.name

        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()[:12]
        return f"{file_path.stem}-{digest}{file_path.suffix}"

    @staticmethod
    def comfy_path_from_upload(upload_response: dict) -> str:
        name = upload_response.get("name") or upload_response.get("filename")
        subfolder = upload_response.get("subfolder", "")
        if not name:
            raise ValueError(f"Unexpected ComfyUI upload response: {upload_response}")
        return f"{subfolder}/{name}" if subfolder else name
