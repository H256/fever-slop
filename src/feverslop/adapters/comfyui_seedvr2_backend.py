from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_render_queue import ComfyUIRenderQueue
from feverslop.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader
from feverslop.adapters.workflow_patcher import WorkflowPatcher


DEFAULT_WORKFLOW_PATH = Path(__file__).resolve().parents[3] / "workflows" / "video_seedvr2_3b_api.json"


@dataclass(frozen=True)
class SeedVR2RenderSettings:
    model: str = "seedvr2_3b_int8_convrot.safetensors"
    vae: str = "seedvr2_ema_vae_fp16.safetensors"
    denoise: float = 0.35
    temporal_overlap: int = 4
    color_correction: str = "lab"
    seed: int = 0
    fps: int = 24
    split_latent: bool = True
    vae_temporal_size: int = 32
    vae_temporal_overlap: int = 8
    trim_start_seconds: float = 0.0
    trim_duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.denoise <= 1.0:
            raise ValueError("denoise must be between 0 and 1")
        if self.temporal_overlap < 0:
            raise ValueError("temporal_overlap must not be negative")
        if self.vae_temporal_size < 8:
            raise ValueError("vae_temporal_size must be at least 8")
        if self.vae_temporal_overlap < 4 or self.vae_temporal_overlap > self.vae_temporal_size:
            raise ValueError("vae_temporal_overlap must be between 4 and vae_temporal_size")
        if self.trim_start_seconds < 0:
            raise ValueError("trim_start_seconds must not be negative")
        if self.trim_duration_seconds is not None and self.trim_duration_seconds <= 0:
            raise ValueError("trim_duration_seconds must be positive")
        if self.color_correction not in {"lab", "wavelet", "adain", "none"}:
            raise ValueError("color_correction must be lab, wavelet, adain, or none")


class ComfyUISeedVR2Backend:
    def __init__(
        self,
        *,
        client: ComfyUIClient,
        asset_uploader: ComfyUIVideoAssetUploader | None = None,
        render_queue: ComfyUIRenderQueue | None = None,
        workflow_path: str | Path = DEFAULT_WORKFLOW_PATH,
    ):
        self.client = client
        self.asset_uploader = asset_uploader or ComfyUIVideoAssetUploader(client)
        self.render_queue = render_queue or ComfyUIRenderQueue(client)
        self.workflow_path = Path(workflow_path)

    def build_workflow(
        self,
        *,
        video_name: str,
        output_prefix: str,
        output_size: tuple[int, int],
        settings: SeedVR2RenderSettings,
    ) -> dict[str, dict[str, Any]]:
        width, height = output_size
        if width <= 0 or height <= 0:
            raise ValueError("output_size must be positive")
        workflow = json.loads(self.workflow_path.read_text(encoding="utf-8-sig"))
        patcher = WorkflowPatcher(workflow)
        patcher.set_input_by_title("#LOAD_VIDEO", "file", video_name)
        trim_enabled = settings.trim_duration_seconds is not None
        patcher.set_input_by_title("#VIDEO_SOURCE", "switch", trim_enabled)
        if trim_enabled:
            patcher.set_input_by_title("#VIDEO_SLICE", "start_time", settings.trim_start_seconds)
            patcher.set_input_by_title("#VIDEO_SLICE", "duration", settings.trim_duration_seconds)
        patcher.set_input_by_title("#RESIZE_VIDEO", "resize_type", "scale dimensions")
        patcher.set_input_by_title("#RESIZE_VIDEO", "resize_type.width", width)
        patcher.set_input_by_title("#RESIZE_VIDEO", "resize_type.height", height)
        patcher.set_input_by_title("#RESIZE_VIDEO", "resize_type.crop", "disabled")
        patcher.set_input_by_title("#SEEDVR_MODEL", "unet_name", settings.model)
        patcher.set_input_by_title("#SEEDVR_VAE", "vae_name", settings.vae)
        patcher.set_input_by_title("#TEMPORAL_CHUNK", "temporal_overlap", settings.temporal_overlap)
        for title in ("#VAE_ENCODE_TILED", "#VAE_DECODE_TILED"):
            patcher.set_input_by_title(title, "temporal_size", settings.vae_temporal_size)
            patcher.set_input_by_title(title, "temporal_overlap", settings.vae_temporal_overlap)
        patcher.set_input_by_title("#SPLIT_LATENT_BOOLEAN", "value", settings.split_latent)
        patcher.set_input_by_title("#SEEDVR_SAMPLER", "seed", settings.seed)
        patcher.set_input_by_title("#SEEDVR_SAMPLER", "denoise", settings.denoise)
        patcher.set_input_by_title("#COLOR_CORRECTION", "color_correction_method", settings.color_correction)
        patcher.set_input_by_title("#SAVE_VIDEO", "filename_prefix", output_prefix)
        return patcher.get()

    def render(
        self,
        *,
        source_video: str | Path,
        output_path: str | Path,
        output_size: tuple[int, int],
        scene_number: int,
        pass_number: int,
        settings: SeedVR2RenderSettings,
        segment_number: int | None = None,
    ) -> Path:
        source_video = Path(source_video)
        output_path = Path(output_path)
        upload = self.client.upload_file_via_image_endpoint(
            source_video,
            subfolder="feverslop/seedvr2/input",
            file_type="input",
            overwrite=True,
            upload_name=ComfyUIVideoAssetUploader.content_addressed_name(source_video),
        )
        video_name = ComfyUIVideoAssetUploader.comfy_path_from_upload(upload)
        output_suffix = f"_segment_{segment_number:04d}" if segment_number is not None else ""
        workflow = self.build_workflow(
            video_name=video_name,
            output_prefix=f"feverslop/seedvr2/scene_{scene_number:04d}/pass_{pass_number:02d}{output_suffix}",
            output_size=output_size,
            settings=settings,
        )
        return self.render_queue.queue_workflow_and_download_first_video(
            workflow,
            scene_number=scene_number,
            output_path=output_path,
        )
