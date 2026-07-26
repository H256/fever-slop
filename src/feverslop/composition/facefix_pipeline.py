from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_facefix_backend import ComfyUIFaceFixRenderBackend
from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver
from feverslop.application.facefix_pipeline import FaceFixPipelineStep, FaceFixRequest
from feverslop.config.app_config import AppConfig
from feverslop.domain.facefix_rendering import FaceFixConfig
from feverslop.path_utils import coerce_local_path
from feverslop.ports.reporting import ConsoleReporter


@dataclass(frozen=True)
class FaceFixCompositionOptions:
    app_config_path: str | Path = "./app_config.json"
    workflow_path: str | Path = ""
    scenes_dir: str | Path = ""
    project_dir: str | Path | None = None
    scene_numbers: list[int] | None = None
    reference_images: list[Path] | None = None
    skip_existing: bool = True
    postprocess: bool = True
    ffmpeg_path: str = "ffmpeg"
    postprocess_reencode: bool = True
    ffmpeg_debug: bool = False
    keyframe_indices: str = "0,16,32,48"
    guiding_strength: float = 0.2
    cond_image_strength: float = 0.5
    temporal_tile_size: int = 56
    temporal_overlap: int = 24
    temporal_overlap_cond_strength: float = 0.5


def build_facefix_step(
    options: FaceFixCompositionOptions,
    *,
    console: Console | None = None,
) -> FaceFixPipelineStep:
    app_config = AppConfig.load(options.app_config_path)
    client = ComfyUIClient(
        base_url=app_config.comfyui.base_url,
        prompt_timeout_seconds=app_config.comfyui.prompt_timeout_seconds,
    )
    model_resolver = ComfyUIModelResolver(
        client,
        overrides=app_config.comfyui.model_overrides,
    )

    config = FaceFixConfig(
        keyframe_indices=options.keyframe_indices,
        guiding_strength=options.guiding_strength,
        cond_image_strength=options.cond_image_strength,
        temporal_tile_size=options.temporal_tile_size,
        temporal_overlap=options.temporal_overlap,
        temporal_overlap_cond_strength=options.temporal_overlap_cond_strength,
        postprocess=options.postprocess,
        ffmpeg_path=options.ffmpeg_path,
    )

    backend = ComfyUIFaceFixRenderBackend(
        client=client,
        workflow_path=coerce_local_path(options.workflow_path),
        project_dir=coerce_local_path(options.project_dir) if options.project_dir else None,
        config=config,
        postprocess=options.postprocess,
        ffmpeg_path=options.ffmpeg_path,
        postprocess_reencode=options.postprocess_reencode,
        ffmpeg_debug=options.ffmpeg_debug,
        model_resolver=model_resolver,
    )

    reporter = ConsoleReporter(console) if console is not None else None
    return FaceFixPipelineStep(
        backend=backend,
        config=config,
        reporter=reporter,
    )


def run_facefix(
    options: FaceFixCompositionOptions,
    *,
    console: Console | None = None,
) -> list[Path]:
    """Build and execute the FaceFix pipeline."""
    step = build_facefix_step(options, console=console)
    scenes_dir = coerce_local_path(options.scenes_dir)

    request = FaceFixRequest(
        scenes_dir=scenes_dir,
        scene_numbers=options.scene_numbers,
        reference_images=options.reference_images or [],
        skip_existing=options.skip_existing,
    )

    return step.execute(request)
