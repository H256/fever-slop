from __future__ import annotations

from pathlib import Path

from rich.console import Console

from feverslop.adapters.comfyui_acestep_song_generator import (
    ComfyUIAceStepSongGenerator,
)
from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver
from feverslop.adapters.full_auto_scaffold import LocalProjectScaffold
from feverslop.adapters.llm_song_brief_generator import LLMSongBriefGenerator
from feverslop.adapters.openai_compatible_llm import OpenAICompatibleLLMClient
from feverslop.adapters.pipeline_runner import RunPipelineAdapter
from feverslop.application.full_auto import FullAutoUseCase
from feverslop.composition import pipeline_runner
from feverslop.config.app_config import AppConfig


def build_full_auto_use_case(
    *,
    app_config_path: str | Path = "app_config.json",
    workflow_path: str | Path | None = None,
    console: Console | None = None,
) -> FullAutoUseCase:
    app_config = AppConfig.load(app_config_path)
    workflow_path = workflow_path or Path(__file__).resolve().parents[3] / "workflows" / "audio" / "audio-model" / "audio_song_v2.json"
    llm = OpenAICompatibleLLMClient(
        base_url=app_config.llm.base_url,
        api_key=app_config.llm.api_key,
        model=app_config.llm.model_for("creative"),
        temperature=app_config.llm.temperature,
        dspy_temperature=app_config.llm.dspy_temperature,
        max_tokens=app_config.llm.max_tokens,
        request_timeout_seconds=app_config.llm.request_timeout_seconds,
        max_concurrent_requests=app_config.llm.max_concurrent_requests,
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
        pipeline_runner=RunPipelineAdapter(
            run_pipeline=pipeline_runner.run,
            build_arg_parser=pipeline_runner.build_arg_parser,
        ),
        console=console,
    )
