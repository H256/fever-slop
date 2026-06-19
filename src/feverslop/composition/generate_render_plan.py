from __future__ import annotations

from pathlib import Path

from rich.console import Console

from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver
from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.adapters.openai_compatible_llm import OpenAICompatibleLLMClient
from feverslop.application.audio_timeline_pipeline import AudioTimelinePipeline
from feverslop.application.generate_render_plan import GenerateRenderPlanUseCase
from feverslop.application.prompt_generation_pipeline import PromptGenerationPipeline
from feverslop.application.render_plan_pipeline import RenderPlanPipeline
from feverslop.application.scene_timeline_pipeline import SceneTimelinePipeline
from feverslop.audio.beat_analysis import BeatImpactAnalyzer
from feverslop.audio.demucs_separator import DemucsSeparator
from feverslop.audio.vocal_timeline_analyzer import VocalTimelineAnalyzer
from feverslop.prompting.lyric_alignment import LyricTimelineAligner
from storyboard_renderer import StoryboardRenderer


def build_generate_render_plan_use_case(console: Console | None = None) -> GenerateRenderPlanUseCase:
    return GenerateRenderPlanUseCase(
        console=console,
        artifact_store=JsonArtifactStore(),
        pipeline_services=[
            AudioTimelinePipeline(
                separator_factory=lambda config: DemucsSeparator(model_name=config.audio.demucs_model),
                vocal_analyzer_factory=_build_vocal_analyzer,
                beat_analyzer_factory=BeatImpactAnalyzer,
                lyric_aligner_factory=lambda context: LyricTimelineAligner(_build_llm(context["app_config"])),
            ),
            SceneTimelinePipeline(),
            PromptGenerationPipeline(llm_factory=_build_llm),
            RenderPlanPipeline(),
        ],
        storyboard_renderer_factory=_build_storyboard_renderer,
    )


def _build_vocal_analyzer(config):
    vocal_cfg = config.vocal_detection
    return VocalTimelineAnalyzer(
        whisper_model=config.audio.whisper_model,
        language=config.audio.language,
        merge_gap=vocal_cfg.merge_gap,
        min_vocal_duration=vocal_cfg.min_vocal_duration,
        min_silence_duration=vocal_cfg.min_silence_duration,
        rms_low_percentile=vocal_cfg.rms_low_percentile,
        rms_high_percentile=vocal_cfg.rms_high_percentile,
        rms_ratio=vocal_cfg.rms_ratio,
        smooth_frames=vocal_cfg.smooth_frames,
    )


def _build_llm(app_config):
    return OpenAICompatibleLLMClient(
        base_url=app_config.llm.base_url,
        model=app_config.llm.model,
        temperature=app_config.llm.temperature,
        max_tokens=app_config.llm.max_tokens,
    )


def _build_storyboard_renderer(app_config, render_dir: Path, workflow_path: Path):
    client = ComfyUIClient(base_url=app_config.comfyui.base_url)
    model_resolver = ComfyUIModelResolver(
        client,
        overrides=app_config.comfyui.model_overrides,
    )
    return StoryboardRenderer(
        client=client,
        zimage_workflow_path=workflow_path,
        output_dir=render_dir / "storyboard",
        positive_prompt_node_title="#POSITIVE_PROMPT",
        negative_prompt_node_title="#NEGATIVE_PROMPT",
        save_image_node_title="#SAVE_IMAGE",
        character_lora_node_title="#LORA_1",
        model_resolver=model_resolver,
    )
