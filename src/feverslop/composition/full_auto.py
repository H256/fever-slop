from __future__ import annotations

from pathlib import Path

from rich.console import Console

from feverslop.adapters.comfyui_acestep_song_generator import ComfyUIAceStepSongGenerator
from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver
from feverslop.adapters.full_auto_scaffold import LocalProjectScaffold
from feverslop.adapters.llm_song_brief_generator import LLMSongBriefGenerator
from feverslop.adapters.openai_compatible_llm import OpenAICompatibleLLMClient
from feverslop.adapters.pipeline_runner import RunPipelineAdapter
from feverslop.application.full_auto import FullAutoUseCase
from feverslop.config.app_config import AppConfig


def build_full_auto_use_case(
    *,
    app_config_path: str | Path = "app_config.json",
    workflow_path: str | Path = Path("workflows") / "audio_song.json",
    console: Console | None = None,
) -> FullAutoUseCase:
    app_config = AppConfig.load(app_config_path)
    llm = OpenAICompatibleLLMClient(
        base_url=app_config.llm.base_url,
        model=app_config.llm.model,
        temperature=app_config.llm.temperature,
        max_tokens=app_config.llm.max_tokens,
    )
    client = ComfyUIClient(
        base_url=app_config.comfyui.base_url,
        prompt_timeout_seconds=app_config.comfyui.prompt_timeout_seconds,
    )
    return FullAutoUseCase(
        brief_generator=LLMSongBriefGenerator(llm),
        song_generator=ComfyUIAceStepSongGenerator(
            client=client,
            workflow_path=workflow_path,
            model_resolver=ComfyUIModelResolver(
                client,
                overrides=app_config.comfyui.model_overrides,
            ),
        ),
        project_scaffold=LocalProjectScaffold(),
        pipeline_runner=RunPipelineAdapter(),
        console=console,
    )
