from __future__ import annotations

from pathlib import Path

from autoprompter.adapters.comfyui_client import ComfyUIClient
from autoprompter.adapters.comfyui_rendering import ComfyUIImageBackend
from autoprompter.adapters.local_artifacts import JsonArtifactStore
from autoprompter.application.render_storyboard import RenderStoryboardUseCase
from autoprompter.config.app_config import AppConfig


def build_render_storyboard_use_case(
    *,
    app_config: AppConfig,
    workflow_path: str | Path,
    output_dir: str | Path,
) -> RenderStoryboardUseCase:
    client = ComfyUIClient(base_url=app_config.comfyui.base_url)
    return RenderStoryboardUseCase(
        backend=ComfyUIImageBackend(
            client=client,
            workflow_path=workflow_path,
            output_dir=output_dir,
        ),
        artifact_store=JsonArtifactStore(),
    )
