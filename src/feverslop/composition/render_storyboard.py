from __future__ import annotations

from pathlib import Path

from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver
from feverslop.adapters.comfyui_rendering import ComfyUIImageBackend
from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.application.render_storyboard import RenderStoryboardUseCase
from feverslop.config.app_config import AppConfig


def build_render_storyboard_use_case(
    *,
    app_config: AppConfig,
    workflow_path: str | Path,
    output_dir: str | Path,
) -> RenderStoryboardUseCase:
    client = ComfyUIClient(base_url=app_config.comfyui.base_url)
    model_resolver = ComfyUIModelResolver(
        client,
        overrides=app_config.comfyui.model_overrides,
    )
    return RenderStoryboardUseCase(
        backend=ComfyUIImageBackend(
            client=client,
            workflow_path=workflow_path,
            output_dir=output_dir,
            model_resolver=model_resolver,
        ),
        artifact_store=JsonArtifactStore(),
    )
