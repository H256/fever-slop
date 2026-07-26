from __future__ import annotations

import argparse
from pathlib import Path
from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.openai_compatible_llm import OpenAICompatibleLLMClient  # noqa: F401
from feverslop.adapters.video_postprocessor import VideoPostProcessor  # noqa: F401
from feverslop.application.msr_prompt_enrichment import enrich_render_plan_with_msr_prompts  # noqa: F401
from feverslop.application.reference_bible import enrich_render_plan_with_reference_sheets  # noqa: F401
from feverslop.config.app_config import AppConfig  # noqa: F401
from feverslop.errors import FeverSlopError
from feverslop.prompting.ltx_prompt_anchor_fixer import LTXPromptAnchorFixer  # noqa: F401
from feverslop.prompting.relay_direction_builder import RelayDirectionBuilder  # noqa: F401
from feverslop.tools.reference_bible import run as render_reference_bible  # noqa: F401
from feverslop.tools.storyboard_page import generate_storyboard_page  # noqa: F401

from .arg_parser import PipelineStage, build_arg_parser
from .config_loader import (
    PipelineRunContext,  # noqa: F401
    PipelineRunResult,
    PipelineRunState,
    build_run_context,
    collect_render_plan_scene_clips,  # noqa: F401
    count_render_plan_items,  # noqa: F401
    convert_to_safe_file_stem,  # noqa: F401
    resolve_runner_path,
    rewrite_concat_list,  # noqa: F401
    runner_root,  # noqa: F401
)
from .generate_render_plan import build_generate_render_plan_use_case  # noqa: F401
from .render_storyboard import build_render_storyboard_use_case  # noqa: F401
from .render_video import build_render_video_scenes_use_case  # noqa: F401
from .stage_runners import (
    RenderProgressReporter,  # noqa: F401
    STAGE_LABELS,
    STAGE_RUNNERS,
    _initial_render_plan,
    console,
    resolve_pipeline_stages,
    run_unittest_suite,  # noqa: F401
    write_step,
)


COMFYUI_RENDERING_STAGES = frozenset({
    PipelineStage.STORYBOARD_FRAMES,
    PipelineStage.MSR_REFERENCES,
    PipelineStage.LTX_RENDER_SCENES,
    PipelineStage.FACEFIX,
})


def build_run_state(args: argparse.Namespace, stages: list[PipelineStage]) -> PipelineRunState:
    context = build_run_context(args)
    app_config_path = resolve_runner_path(args.app_config)
    state = PipelineRunState(
        args=args,
        context=context,
        app_config_path=app_config_path,
        storyboard_workflow=resolve_runner_path(args.storyboard_workflow),
        reference_hero_workflow=resolve_runner_path(args.reference_hero_workflow),
        reference_edit_workflow=resolve_runner_path(args.reference_edit_workflow),
        msr_workflow=resolve_runner_path(args.msr_workflow),
        ingredients_workflow=resolve_runner_path(args.ingredients_workflow),
        relay_workflow=resolve_runner_path(args.relay_workflow) if str(args.relay_workflow).strip() else Path(""),
        single_prompt_workflow=resolve_runner_path(args.single_prompt_workflow),
        facefix_workflow=resolve_runner_path(args.facefix_workflow),
        facefix_crop_workflow=resolve_runner_path(args.facefix_crop_workflow),
        plan_for_next_step=_initial_render_plan(context, args, stages),
    )
    app_config = AppConfig.load(app_config_path)
    state.comfyui_client = ComfyUIClient(
        base_url=app_config.comfyui.base_url,
        prompt_timeout_seconds=app_config.comfyui.prompt_timeout_seconds,
    )
    console.print(f"Project: {context.project_config_path}")
    console.print(f"Input audio: {context.input_audio}")
    console.print(f"Song ID: {context.song_id}")
    console.print(f"Render mode: {args.render_mode}")
    console.print("Stages: " + ", ".join(stage.value for stage in stages))
    return state


def run(args: argparse.Namespace) -> PipelineRunResult:
    stages = resolve_pipeline_stages(args)
    state = build_run_state(args, stages)
    for stage in stages:
        if stage in COMFYUI_RENDERING_STAGES and state.comfyui_client is not None:
            console.print(f"[dim]Clearing ComfyUI cache and VRAM before {STAGE_LABELS[stage]}...[/dim]")
            state.comfyui_client.free_cache_and_vram()
        write_step(f"Stage {STAGE_LABELS[stage]}")
        try:
            STAGE_RUNNERS[stage](state)
        except FeverSlopError as exc:
            console.print(f"[red]Pipeline error:[/red] {exc}")
            raise
        except Exception as exc:
            raise RuntimeError(f"{STAGE_LABELS[stage]} failed: {exc}") from exc

    console.print("Pipeline complete.")
    console.print(f"Render plan: {state.plan_for_next_step}")
    if state.final_video_path:
        console.print(f"Final video: {state.final_video_path}")
    elif state.video_only_path:
        console.print(f"Video-only concat: {state.video_only_path}")

    return PipelineRunResult(
        render_plan_path=state.plan_for_next_step,
        final_video_path=state.final_video_path,
        video_only_path=state.video_only_path,
    )


def main() -> None:
    run(build_arg_parser().parse_args())
