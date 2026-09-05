from __future__ import annotations

import random
from pathlib import Path

from rich.console import Console

from feverslop.composition.continuation_capability import resolve_continuation_capability

from feverslop.adapters.audio.beat_analysis import (
    BeatImpactAnalyzer,
    BeatSceneDurationGenerator,
)
from feverslop.adapters.audio.demucs_separator import DemucsSeparator
from feverslop.adapters.audio.vocal_timeline_analyzer import VocalTimelineAnalyzer
from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver
from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.adapters.h3_prompt_checkpoints import H3PromptCheckpointStore
from feverslop.adapters.openai_compatible_llm import OpenAICompatibleLLMClient
from feverslop.adapters.storyboard_renderer import StoryboardRenderer
from feverslop.application.audio_timeline_pipeline import AudioTimelinePipeline
from feverslop.application.generate_render_plan import (
    GenerateRenderPlanExecutionRequest,
    GenerateRenderPlanRequest,
    GenerateRenderPlanResult,
    GenerateRenderPlanUseCase,
)
from feverslop.application.h3_prompt_pipeline import H3PromptPipeline
from feverslop.application.prompt_generation_pipeline import PromptGenerationPipeline
from feverslop.application.render_plan_pipeline import RenderPlanPipeline
from feverslop.application.scene_timeline_pipeline import SceneTimelinePipeline
from feverslop.config.app_config import AppConfig
from feverslop.config.project_config import ProjectConfig, ProjectPaths
from feverslop.composition.canonical_plan_regenerator import CanonicalPlanRegenerator
from feverslop.domain.ltx_rendering import resolve_rolling_frame_profile
from feverslop.domain.scene_duration_limits import resolve_scene_duration_policy
from feverslop.domain.timeline_transform import (
    merge_same_kind_segments,
    normalize_empty_vocals,
)
from feverslop.pipeline.prompt_relay_builder import build_scene_prompt_relay
from feverslop.pipeline.render_plan_builder import build_render_plan
from feverslop.pipeline.scene_duration_enforcer import (
    enforce_scene_srt_file,
    parse_srt_scenes,
    validate_scene_durations,
)
from feverslop.pipeline.stage1_segment_builder import build_stage1_segment_json
from feverslop.pipeline.utils import save_timeline_json
from feverslop.prompting.concept_prompt_batcher import ConceptPromptBatcher
from feverslop.prompting.dspy_h3_prompt_builder import (
    DspyH3PromptBuilder,
    build_dspy_generator,
)
from feverslop.prompting.h3_prompt_builder import H3PromptBuilder
from feverslop.prompting.lyric_alignment import LyricTimelineAligner
from feverslop.prompting.prompt_pipeline import MusicVideoPromptPipeline
from feverslop.prompting.scene_prompt_builder import ScenePromptBuilder


def _common_pipeline_services():
    return [
        SceneTimelinePipeline(
            scene_generator_factory=_build_scene_generator,
            enforce_scene_srt_file=enforce_scene_srt_file,
            parse_scene_srt=parse_srt_scenes,
            validate_scene_durations=validate_scene_durations,
            build_stage1_segment_json=build_stage1_segment_json,
            build_scene_prompt_relay=build_scene_prompt_relay,
        ),
        PromptGenerationPipeline(
            llm_factory=_build_llm,
            prompt_pipeline_factory=MusicVideoPromptPipeline,
            concept_batcher_factory=ConceptPromptBatcher,
            scene_prompt_builder_factory=ScenePromptBuilder,
        ),
        H3PromptPipeline(
            llm_factory=_build_llm,
            h3_prompt_builder_factory=H3PromptBuilder,
            dspy_prompt_builder_factory=lambda llm: DspyH3PromptBuilder(
                build_dspy_generator(llm), allow_fallback=True,
            ),
            checkpoint_store_factory=_build_h3_checkpoint_store,
        ),
        RenderPlanPipeline(
            build_render_plan=build_render_plan,
            regenerator_factory=_build_canonical_regenerator,
        ),
    ]


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
                normalize_empty_vocals=normalize_empty_vocals,
                merge_same_kind_segments=merge_same_kind_segments,
                save_timeline_json=save_timeline_json,
            ),
            *_common_pipeline_services(),
        ],
        storyboard_renderer_factory=_build_storyboard_renderer,
    )


def build_generate_render_plan_execution_request(
    request: GenerateRenderPlanRequest,
    *,
    resolution: tuple[int, int] | None = None,
) -> GenerateRenderPlanExecutionRequest:
    config = ProjectConfig.load(request.project_config_path)
    if resolution is not None:
        config = config.apply_resolution_override(width=resolution[0], height=resolution[1])
    paths = ProjectPaths.from_config(config)
    app_config = AppConfig.load(request.app_config_path)
    video_settings = config.to_video_settings()
    profile = app_config.resolve_video_workflow_profile(
        pipeline=config.video_pipeline,
        purpose="final",
    )
    duration_capability = resolve_continuation_capability(config.video_pipeline, profile)
    preroll_frames, tail_frames, round_render_frames_to_8n1 = resolve_rolling_frame_profile(
        request.rolling_frame_profile,
        video_pipeline=config.video_pipeline,
    )
    workflow_limits = {
        limit.workflow: limit.max_render_duration_seconds
        for limit in app_config.comfyui.video_workflow_limits
    }
    scene_duration_policy = resolve_scene_duration_policy(
        requested_min_seconds=config.scene_generation.min_duration,
        requested_max_seconds=config.scene_generation.max_duration,
        fps=video_settings.fps,
        preroll_frames=preroll_frames,
        tail_frames=tail_frames,
        round_render_frames_to_8n1=round_render_frames_to_8n1,
        workflow_limits=workflow_limits,
        workflow_paths=request.video_workflow_paths,
        default_max_render_duration_seconds=(
            app_config.comfyui.default_max_render_duration_seconds
        ),
        duration_capability=duration_capability,
        allow_continuation=config.video_pipeline == "minimax-h3-r2v",
    )
    song_id = getattr(config, "song_id", None) or getattr(config, "project_name", "") or config.input_audio.stem
    return GenerateRenderPlanExecutionRequest(
        source_request=request,
        config=config,
        paths=paths,
        app_config=app_config,
        video_settings=video_settings,
        song_id=song_id,
        scene_duration_policy=scene_duration_policy,
        duration_capability=duration_capability,
    )


def execute_generate_render_plan(
    request: GenerateRenderPlanRequest,
    *,
    console: Console | None = None,
    resolution: tuple[int, int] | None = None,
) -> GenerateRenderPlanResult:
    use_case = build_generate_render_plan_use_case(console=console)
    return use_case.execute(build_generate_render_plan_execution_request(request, resolution=resolution))


def build_rebuild_render_plan_use_case(console: Console | None = None) -> GenerateRenderPlanUseCase:
    """Build a use case that rebuilds render plan without re-analyzing audio.

    Skips the AudioTimelinePipeline step, using existing timeline/beat JSON from disk.
    Intended for timeline editing workflows where audio analysis is already done.
    """
    return GenerateRenderPlanUseCase(
        console=console,
        artifact_store=JsonArtifactStore(),
        pipeline_services=_common_pipeline_services(),
        storyboard_renderer_factory=_build_storyboard_renderer,
    )


def execute_rebuild_render_plan(
    request: GenerateRenderPlanRequest,
    *,
    console: Console | None = None,
) -> GenerateRenderPlanResult:
    """Rebuild render plan from existing timeline data, skipping audio analysis."""
    use_case = build_rebuild_render_plan_use_case(console=console)
    return use_case.execute(build_generate_render_plan_execution_request(request))


def _build_canonical_regenerator(context) -> CanonicalPlanRegenerator:
    config = context["config"]
    paths = context["paths"]
    return CanonicalPlanRegenerator(
        config.project_dir,
        reference_plan_path=paths.artifact_layout.references_plan,
        reporter=context["reporter"],
    )


def _build_h3_checkpoint_store(context) -> H3PromptCheckpointStore:
    return H3PromptCheckpointStore(
        context["config"].project_dir,
        reporter=context["reporter"],
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
        api_key=app_config.llm.api_key,
        model=app_config.llm.model_for("structured"),
        temperature=app_config.llm.temperature,
        dspy_temperature=app_config.llm.dspy_temperature,
        max_tokens=app_config.llm.max_tokens,
        request_timeout_seconds=app_config.llm.request_timeout_seconds,
        dspy_cache=app_config.llm.dspy_cache,
        max_concurrent_requests=app_config.llm.max_concurrent_requests,
        prompt_judge_attempts=app_config.llm.prompt_judge_attempts,
        prompt_judge_max_tokens=app_config.llm.prompt_judge_max_tokens,
        prompt_judge_blocking=app_config.llm.prompt_judge_blocking,
        chat_template_kwargs=app_config.llm.chat_template_kwargs,
    )


def _build_scene_generator(scene_cfg):
    seed = int(scene_cfg.seed)
    if seed == -1:
        seed = random.SystemRandom().randint(0, 2**31 - 1)
    return BeatSceneDurationGenerator(
        min_duration=scene_cfg.min_duration,
        max_duration=scene_cfg.max_duration,
        bias=scene_cfg.bias,
        duration_preset=scene_cfg.duration_preset,
        seed=seed,
    )


def _build_storyboard_renderer(app_config, render_dir: Path, workflow_path: Path):
    client = ComfyUIClient(
        base_url=app_config.comfyui.base_url,
        prompt_timeout_seconds=app_config.comfyui.prompt_timeout_seconds,
    )
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
