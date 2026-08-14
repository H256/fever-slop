from __future__ import annotations

from pathlib import Path

from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver
from feverslop.adapters.comfyui_rendering import ComfyUIImageBackend
from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.adapters.openai_compatible_llm import OpenAICompatibleLLMClient
from feverslop.application.render_storyboard import RenderStoryboardUseCase
from feverslop.config.app_config import AppConfig, StoryboardPromptTransformConfig
from feverslop.path_utils import coerce_local_path
from feverslop.prompting.storyboard_prompt_transformer import TemplateStoryboardPromptTransformer


def build_render_storyboard_use_case(
    *,
    app_config: AppConfig,
    workflow_path: str | Path,
    output_dir: str | Path,
) -> RenderStoryboardUseCase:
    client = ComfyUIClient(
        base_url=app_config.comfyui.base_url,
        prompt_timeout_seconds=app_config.comfyui.prompt_timeout_seconds,
    )
    model_resolver = ComfyUIModelResolver(
        client,
        overrides=app_config.comfyui.model_overrides,
    )
    transform_config = matching_storyboard_prompt_transform(app_config, workflow_path)
    prompt_transformer = None
    positive_prompt_input = None
    if transform_config is not None:
        if transform_config.kind != "template":
            raise ValueError(f"Unsupported storyboard prompt transform kind: {transform_config.kind}")
        if not transform_config.template:
            raise ValueError("Storyboard prompt transform template is required")
        llm = OpenAICompatibleLLMClient(
            base_url=app_config.llm.base_url,
            api_key=app_config.llm.api_key,
            model=app_config.llm.model,
            temperature=app_config.llm.temperature,
            dspy_temperature=app_config.llm.dspy_temperature,
            max_tokens=app_config.llm.max_tokens,
            request_timeout_seconds=app_config.llm.request_timeout_seconds,
            max_concurrent_requests=app_config.llm.max_concurrent_requests,
        )
        prompt_transformer = TemplateStoryboardPromptTransformer(
            llm=llm,
            template_path=coerce_local_path(transform_config.template),
            debug_dir=Path(output_dir) / transform_config.debug_dir,
        )
        positive_prompt_input = transform_config.positive_prompt_input

    return RenderStoryboardUseCase(
        backend=ComfyUIImageBackend(
            client=client,
            workflow_path=workflow_path,
            output_dir=output_dir,
            model_resolver=model_resolver,
        ),
        artifact_store=JsonArtifactStore(),
        prompt_transformer=prompt_transformer,
        positive_prompt_input=positive_prompt_input,
    )


def matching_storyboard_prompt_transform(
    app_config: AppConfig,
    workflow_path: str | Path,
) -> StoryboardPromptTransformConfig | None:
    normalized_path = normalize_workflow_path(str(workflow_path))
    for transform in app_config.storyboard_prompt_transforms:
        transform_path = normalize_workflow_path(transform.workflow)
        if normalized_path == transform_path or normalized_path.endswith(f"/{transform_path}"):
            return transform
    return None


def normalize_workflow_path(value: str) -> str:
    return value.replace("\\", "/").casefold()
