from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.comfyui_minimax_h3_r2v_backend import ComfyUIMiniMaxH3R2VBackend
from feverslop.adapters.comfyui_minimax_h3_t2v_backend import ComfyUIMiniMaxH3T2VBackend
from feverslop.adapters.comfyui_minimax_h3_i2v_backend import ComfyUIMiniMaxH3I2VBackend
from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver
from feverslop.adapters.comfyui_ingredients_video_backend import ComfyUIIngredientsVideoRenderBackend
from feverslop.adapters.comfyui_msr_video_backend import ComfyUIMSRVideoRenderBackend
from feverslop.adapters.comfyui_video_backend import ComfyUIVideoRenderBackend
from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.application.render_video import RenderVideoScenesUseCase
from feverslop.config.app_config import AppConfig
from feverslop.config.project_config import ProjectConfig
from feverslop.domain.ltx_rendering import ROLLING_FRAME_PROFILES  # noqa: F401
from feverslop.domain.ltx_rendering import resolve_rolling_frame_profile
from feverslop.domain.scene_duration_limits import resolve_scene_duration_policy
from feverslop.path_utils import coerce_local_path


@dataclass(frozen=True)
class RenderVideoCompositionOptions:
    app_config_path: str | Path = "./app_config.json"
    project_config_path: str | Path | None = None
    render_plan_path: str | Path | None = None
    workflow_path: str | Path = ""
    output_dir: str | Path = ""
    video_pipeline: str = "ltx_i2v"
    video_workflow_profile: str | None = None
    single_prompt_workflow_path: str | Path | None = None
    render_mode: str = "single_prompt"
    single_prompt_title: str = "#PROMPT"
    single_prompt_input: str = "text"
    character_lora_strength: float | None = None
    lora_1_enabled: bool | None = None
    lora_1_name: str | None = None
    lora_1_strength_model: float | None = None
    lora_1_strength_clip: float | None = None
    lora_split_enabled: bool | None = None
    randomize_seed: bool = False
    seed_offset: int = 100000
    segment_length_mode: str = "frames_minus_one"
    min_duration: float | None = None
    max_duration: float | None = None
    allow_out_of_range_clips: bool = False
    debug_workflows_dir: str | Path | None = None
    rolling_frame_profile: str = "original"
    resolution: tuple[int, int] | None = None
    preroll_frames: int | None = None
    tail_loss_frames: int | None = None
    postprocess: bool = True
    ffmpeg_path: str = "ffmpeg"
    postprocess_reencode: bool = True
    ffmpeg_debug: bool = False
    minimax_audio_ref_stems: str | None = None


def build_render_video_scenes_use_case(
    options: RenderVideoCompositionOptions,
    *,
    console: Console | None = None,
) -> RenderVideoScenesUseCase:
    app_config = AppConfig.load(options.app_config_path)
    workflow_path = _resolve_video_workflow_path(app_config, options)
    resolved = resolve_project_config_defaults(options)
    preroll_frames, tail_loss_frames, round_render_frames_to_8n1 = resolve_rolling_frames(options)
    project_config = resolved["project_config"]
    video_settings = project_config.to_video_settings() if project_config else None
    max_render_frames, max_render_duration_seconds, render_budget_workflow_path = _resolve_render_frame_budget(
        app_config=app_config,
        project_config=project_config,
        workflow_path=workflow_path,
        single_prompt_workflow_path=options.single_prompt_workflow_path,
        render_mode=options.render_mode,
        preroll_frames=preroll_frames,
        tail_loss_frames=tail_loss_frames,
        round_render_frames_to_8n1=round_render_frames_to_8n1,
    )

    client = ComfyUIClient(
        base_url=app_config.comfyui.base_url,
        prompt_timeout_seconds=app_config.comfyui.prompt_timeout_seconds,
    )
    model_resolver = ComfyUIModelResolver(
        client,
        overrides=app_config.comfyui.model_overrides,
    )

    if options.video_pipeline == "ltx_msr":
        project_config_path = options.project_config_path or discover_project_config_path(options.render_plan_path or "")
        project_dir = ProjectConfig.load(project_config_path).project_dir if project_config_path else None
        backend = ComfyUIMSRVideoRenderBackend(
            client=client,
            workflow_path=coerce_local_path(workflow_path),
            output_dir=coerce_local_path(options.output_dir),
            project_dir=project_dir,
            seed_offset=options.seed_offset,
            randomize_seed=options.randomize_seed,
            debug_workflows_dir=coerce_local_path(options.debug_workflows_dir) if options.debug_workflows_dir else None,
            preroll_frames=preroll_frames,
            tail_loss_frames=tail_loss_frames,
            round_render_frames_to_8n1=round_render_frames_to_8n1,
            postprocess=options.postprocess,
            ffmpeg_path=options.ffmpeg_path,
            postprocess_reencode=options.postprocess_reencode,
            ffmpeg_debug=options.ffmpeg_debug,
            model_resolver=model_resolver,
            video_settings=video_settings,
            max_render_frames=max_render_frames,
            max_render_duration_seconds=max_render_duration_seconds,
            render_budget_workflow_path=render_budget_workflow_path,
            workflow_profile=str(
                options.video_workflow_profile or Path(workflow_path).stem
            ),
        )
    elif options.video_pipeline == "ltx_ingredients":
        project_config_path = options.project_config_path or discover_project_config_path(options.render_plan_path or "")
        project_dir = ProjectConfig.load(project_config_path).project_dir if project_config_path else None
        backend = ComfyUIIngredientsVideoRenderBackend(
            client=client,
            workflow_path=coerce_local_path(workflow_path),
            output_dir=coerce_local_path(options.output_dir),
            project_dir=project_dir,
            seed_offset=options.seed_offset,
            randomize_seed=options.randomize_seed,
            debug_workflows_dir=coerce_local_path(options.debug_workflows_dir) if options.debug_workflows_dir else None,
            preroll_frames=preroll_frames,
            tail_loss_frames=tail_loss_frames,
            round_render_frames_to_8n1=round_render_frames_to_8n1,
            postprocess=options.postprocess,
            ffmpeg_path=options.ffmpeg_path,
            postprocess_reencode=options.postprocess_reencode,
            ffmpeg_debug=options.ffmpeg_debug,
            model_resolver=model_resolver,
            video_settings=video_settings,
            max_render_frames=max_render_frames,
            max_render_duration_seconds=max_render_duration_seconds,
            render_budget_workflow_path=render_budget_workflow_path,
            workflow_profile=str(
                options.video_workflow_profile or Path(workflow_path).stem
            ),
        )
    elif options.video_pipeline == "minimax-h3-r2v":
        project_config_path = options.project_config_path or discover_project_config_path(options.render_plan_path or "")
        project_config = ProjectConfig.load(project_config_path) if project_config_path else None
        project_dir = project_config.project_dir if project_config else None
        stem_list: list[str] | None = None
        if options.minimax_audio_ref_stems:
            stem_list = [s.strip() for s in options.minimax_audio_ref_stems.split(",") if s.strip()]
        elif project_config is not None:
            stem_list = list(project_config.minimax_h3_audio_refs.stems)
        backend = ComfyUIMiniMaxH3R2VBackend(
            client=client,
            workflow_path=coerce_local_path(workflow_path),
            output_dir=coerce_local_path(options.output_dir),
            project_dir=project_dir,
            seed_offset=options.seed_offset,
            randomize_seed=options.randomize_seed,
            debug_workflows_dir=coerce_local_path(options.debug_workflows_dir) if options.debug_workflows_dir else None,
            preroll_frames=0,
            tail_loss_frames=0,
            postprocess=options.postprocess,
            ffmpeg_path=options.ffmpeg_path,
            postprocess_reencode=options.postprocess_reencode,
            ffmpeg_debug=options.ffmpeg_debug,
            model_resolver=model_resolver,
            video_settings=video_settings,
            audio_ref_stems=stem_list,
            input_audio=project_config.input_audio if project_config is not None else None,
        )
    elif options.video_pipeline in {"minimax-h3-t2v", "minimax-h3-i2v"}:
        project_config_path = options.project_config_path or discover_project_config_path(options.render_plan_path or "")
        project_dir = ProjectConfig.load(project_config_path).project_dir if project_config_path else None
        backend_class = (
            ComfyUIMiniMaxH3I2VBackend
            if options.video_pipeline == "minimax-h3-i2v"
            else ComfyUIMiniMaxH3T2VBackend
        )
        backend = backend_class(
            client=client,
            workflow_path=coerce_local_path(workflow_path),
            output_dir=coerce_local_path(options.output_dir),
            project_dir=project_dir,
            seed_offset=options.seed_offset,
            randomize_seed=options.randomize_seed,
            debug_workflows_dir=coerce_local_path(options.debug_workflows_dir) if options.debug_workflows_dir else None,
            preroll_frames=0,
            tail_loss_frames=0,
            postprocess=options.postprocess,
            ffmpeg_path=options.ffmpeg_path,
            postprocess_reencode=options.postprocess_reencode,
            ffmpeg_debug=options.ffmpeg_debug,
            model_resolver=model_resolver,
            video_settings=video_settings,
        )
    else:
        backend = ComfyUIVideoRenderBackend(
            client=client,
            ltx_workflow_path=coerce_local_path(workflow_path),
            output_dir=coerce_local_path(options.output_dir),
            single_prompt_workflow_path=(
                coerce_local_path(options.single_prompt_workflow_path)
                if options.single_prompt_workflow_path
                else None
            ),
            render_mode=options.render_mode,
            single_prompt_node_title=options.single_prompt_title,
            single_prompt_input_name=options.single_prompt_input,
            character_lora_strength=options.character_lora_strength,
            lora_1_enabled=resolved["lora_1_enabled"],
            lora_1_name=resolved["lora_1_name"],
            lora_1_strength_model=resolved["lora_1_strength_model"],
            lora_1_strength_clip=resolved["lora_1_strength_clip"],
            lora_1_strengths_explicit=resolved["lora_1_strengths_explicit"],
            loras=resolved["loras"],
            lora_split_enabled=resolved["lora_split_enabled"],
            randomize_seed=options.randomize_seed,
            seed_offset=options.seed_offset,
            segment_length_mode=options.segment_length_mode,
            min_duration=resolved["min_duration"],
            max_duration=resolved["max_duration"],
            allow_out_of_range_clips=options.allow_out_of_range_clips,
            debug_workflows_dir=coerce_local_path(options.debug_workflows_dir) if options.debug_workflows_dir else None,
            preroll_frames=preroll_frames,
            tail_loss_frames=tail_loss_frames,
            round_render_frames_to_8n1=round_render_frames_to_8n1,
            postprocess=options.postprocess,
            ffmpeg_path=options.ffmpeg_path,
            postprocess_reencode=options.postprocess_reencode,
            ffmpeg_debug=options.ffmpeg_debug,
            model_resolver=model_resolver,
            video_settings=video_settings,
            max_render_frames=max_render_frames,
            max_render_duration_seconds=max_render_duration_seconds,
            render_budget_workflow_path=render_budget_workflow_path,
        )

    return RenderVideoScenesUseCase(
        backend=backend,
        artifact_store=JsonArtifactStore(),
        console=console,
    )


def discover_project_config_path(render_plan_path: str | Path) -> Path | None:
    render_plan_path = coerce_local_path(render_plan_path).resolve()
    for parent in render_plan_path.parents:
        candidate = parent / "config.json"
        if candidate.exists():
            return candidate
    return None


def _resolve_video_workflow_path(
    app_config: AppConfig,
    options: RenderVideoCompositionOptions,
) -> str | Path:
    if options.workflow_path or options.video_pipeline != "ltx_ingredients":
        return options.workflow_path
    profile = app_config.resolve_video_workflow_profile(
        pipeline=options.video_pipeline,
        purpose="final",
        name=options.video_workflow_profile,
    )
    return profile.workflow_path if profile else options.workflow_path


def resolve_project_config_defaults(options: RenderVideoCompositionOptions) -> dict:
    project_config_path = options.project_config_path
    if not project_config_path and options.render_plan_path:
        discovered = discover_project_config_path(options.render_plan_path)
        if discovered:
            project_config_path = discovered

    project_config = ProjectConfig.load(project_config_path) if project_config_path else None
    if project_config is not None and options.resolution is not None:
        project_config = project_config.apply_resolution_override(
            width=options.resolution[0],
            height=options.resolution[1],
        )
    scene_generation = project_config.scene_generation if project_config else None
    lora_1 = project_config.lora_1 if project_config else None
    loras = _resolve_loras(options, project_config)
    first_lora = loras[0] if loras else None

    min_duration = options.min_duration if options.min_duration is not None else (
        scene_generation.min_duration if scene_generation else 2.0
    )
    max_duration = options.max_duration if options.max_duration is not None else (
        scene_generation.max_duration if scene_generation else 10.0
    )

    lora_1_enabled = options.lora_1_enabled if options.lora_1_enabled is not None else (
        first_lora.enabled if first_lora else (lora_1.enabled if lora_1 else False)
    )
    lora_1_name = options.lora_1_name if options.lora_1_name is not None else (
        first_lora.name if first_lora else (lora_1.name if lora_1 else "")
    )
    lora_1_strength_model = (
        options.lora_1_strength_model
        if options.lora_1_strength_model is not None
        else (first_lora.strength_model if first_lora else (lora_1.strength_model if lora_1 else 1.0))
    )
    lora_1_strength_clip = (
        options.lora_1_strength_clip
        if options.lora_1_strength_clip is not None
        else (first_lora.strength_clip if first_lora else (lora_1.strength_clip if lora_1 else 1.0))
    )
    lora_1_strengths_explicit = (
        options.lora_1_strength_model is not None
        or options.lora_1_strength_clip is not None
    )
    lora_split_enabled = (
        options.lora_split_enabled
        if options.lora_split_enabled is not None
        else (project_config.lora_split_enabled if project_config else False)
    )

    return {
        "project_config": project_config,
        "min_duration": float(min_duration),
        "max_duration": float(max_duration),
        "lora_1_enabled": bool(lora_1_enabled),
        "lora_1_name": str(lora_1_name or ""),
        "lora_1_strength_model": float(lora_1_strength_model),
        "lora_1_strength_clip": float(lora_1_strength_clip),
        "lora_1_strengths_explicit": bool(lora_1_strengths_explicit),
        "loras": loras,
        "lora_split_enabled": bool(lora_split_enabled),
    }


def resolve_rolling_frames(options: RenderVideoCompositionOptions) -> tuple[int, int, bool]:
    profile_preroll, profile_tail, profile_rounding = resolve_rolling_frame_profile(
        options.rolling_frame_profile,
        video_pipeline=options.video_pipeline,
    )
    preroll = profile_preroll if options.preroll_frames is None else options.preroll_frames
    tail = profile_tail if options.tail_loss_frames is None else options.tail_loss_frames
    return max(0, int(preroll)), max(0, int(tail)), bool(profile_rounding)


def _resolve_render_frame_budget(
    *,
    app_config: AppConfig,
    project_config: ProjectConfig | None,
    workflow_path: str | Path,
    single_prompt_workflow_path: str | Path | None,
    render_mode: str,
    preroll_frames: int,
    tail_loss_frames: int,
    round_render_frames_to_8n1: bool,
) -> tuple[int | None, float | None, str | None]:
    if render_mode == "single_prompt" and single_prompt_workflow_path:
        workflow_paths = [single_prompt_workflow_path]
    else:
        workflow_paths = [workflow_path]
    if render_mode == "auto" and single_prompt_workflow_path:
        workflow_paths.append(single_prompt_workflow_path)
    if project_config is None:
        duration, limiting_workflow = _resolve_configured_workflow_duration(
            app_config, workflow_paths
        )
        return None, duration, limiting_workflow

    policy = resolve_scene_duration_policy(
        requested_min_seconds=project_config.scene_generation.min_duration,
        requested_max_seconds=project_config.scene_generation.max_duration,
        fps=project_config.to_video_settings().fps,
        preroll_frames=preroll_frames,
        tail_frames=tail_loss_frames,
        round_render_frames_to_8n1=round_render_frames_to_8n1,
        workflow_limits={
            limit.workflow: limit.max_render_duration_seconds
            for limit in app_config.comfyui.video_workflow_limits
        },
        workflow_paths=workflow_paths,
        default_max_render_duration_seconds=(
            app_config.comfyui.default_max_render_duration_seconds
        ),
    )
    return (
        policy.max_render_frames,
        policy.max_render_duration_seconds,
        policy.limiting_workflow,
    )


def _resolve_configured_workflow_duration(
    app_config: AppConfig,
    workflow_paths: list[str | Path],
) -> tuple[float | None, str | None]:
    limits = {
        Path(limit.workflow).name.casefold(): limit.max_render_duration_seconds
        for limit in app_config.comfyui.video_workflow_limits
    }
    candidates: list[tuple[float, str]] = []
    for workflow_path in workflow_paths:
        basename = Path(str(workflow_path).strip()).name
        duration = limits.get(
            basename.casefold(),
            app_config.comfyui.default_max_render_duration_seconds,
        )
        if duration is not None:
            candidates.append((duration, basename))
    if not candidates:
        return None, None
    duration, workflow = min(candidates, key=lambda item: item[0])
    return float(duration), workflow


def namespace_to_options(args) -> RenderVideoCompositionOptions:
    return RenderVideoCompositionOptions(
        app_config_path=args.app_config,
        project_config_path=getattr(args, "project_config", None),
        render_plan_path=args.render_plan,
        workflow_path=args.workflow,
        output_dir=args.output_dir,
        video_pipeline=getattr(args, "video_pipeline", "ltx_i2v"),
        video_workflow_profile=getattr(args, "video_workflow_profile", None),
        single_prompt_workflow_path=args.single_prompt_workflow,
        render_mode=args.render_mode,
        single_prompt_title=args.single_prompt_title,
        single_prompt_input=args.single_prompt_input,
        character_lora_strength=args.character_lora_strength,
        lora_1_enabled=args.lora_1_enabled,
        lora_1_name=args.lora_1_name,
        lora_1_strength_model=args.lora_1_strength_model,
        lora_1_strength_clip=args.lora_1_strength_clip,
        lora_split_enabled=args.lora_split_enabled,
        randomize_seed=args.randomize_seed,
        seed_offset=args.seed_offset,
        segment_length_mode=args.segment_length_mode,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        allow_out_of_range_clips=args.allow_out_of_range_clips,
        debug_workflows_dir=args.debug_workflows_dir,
        rolling_frame_profile=args.rolling_frame_profile,
        resolution=getattr(args, "resolution", None),
        preroll_frames=args.preroll_frames,
        tail_loss_frames=args.tail_loss_frames,
        postprocess=not args.no_postprocess,
        ffmpeg_path=args.ffmpeg,
        postprocess_reencode=not args.postprocess_streamcopy,
        ffmpeg_debug=getattr(args, "debug", False),
    )


def _resolve_loras(options: RenderVideoCompositionOptions, project_config: ProjectConfig | None):
    from feverslop.adapters.ltx_workflow_patcher import ResolvedLoraConfig

    loras = [
        ResolvedLoraConfig(
            index=index,
            enabled=lora.enabled,
            name=lora.name,
            strength_model=lora.strength_model,
            strength_clip=lora.strength_clip,
            name_explicit=lora.name_explicit,
            strength_model_explicit=lora.strength_model_explicit,
            strength_clip_explicit=lora.strength_clip_explicit,
        )
        for index, lora in enumerate(project_config.loras, start=1)
    ] if project_config else []

    has_lora_1_override = (
        options.lora_1_enabled is not None
        or options.lora_1_name is not None
        or options.lora_1_strength_model is not None
        or options.lora_1_strength_clip is not None
    )
    if has_lora_1_override:
        while len(loras) < 1:
            loras.append(ResolvedLoraConfig(index=1))
        current = loras[0]
        loras[0] = ResolvedLoraConfig(
            index=1,
            enabled=options.lora_1_enabled if options.lora_1_enabled is not None else current.enabled,
            name=options.lora_1_name if options.lora_1_name is not None else current.name,
            name_explicit=(
                bool(str(options.lora_1_name).strip())
                if options.lora_1_name is not None
                else current.name_explicit
            ),
            strength_model=(
                options.lora_1_strength_model
                if options.lora_1_strength_model is not None
                else current.strength_model
            ),
            strength_model_explicit=(
                True
                if options.lora_1_strength_model is not None
                else current.strength_model_explicit
            ),
            strength_clip=(
                options.lora_1_strength_clip
                if options.lora_1_strength_clip is not None
                else current.strength_clip
            ),
            strength_clip_explicit=(
                True
                if options.lora_1_strength_clip is not None
                else current.strength_clip_explicit
            ),
        )

    return tuple(loras)
