from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.adapters.pipeline_runner_options import default_single_prompt_workflow
from feverslop.config.app_config import AppConfig
from feverslop.errors import FeverSlopError

from .arg_parser import PipelineStage, build_arg_parser
from .config_loader import (
    PipelineRunResult,
    PipelineRunState,
    build_run_context,
    resolve_runner_path,
)
from .stage_runners import (
    STAGE_LABELS,
    STAGE_RUNNERS,
    _initial_render_plan,
    console,
    resolve_pipeline_stages,
    write_step,
)

COMFYUI_RENDERING_STAGES = frozenset({
    PipelineStage.STORYBOARD_FRAMES,
    PipelineStage.MSR_REFERENCES,
    PipelineStage.LTX_RENDER_SCENES,
    PipelineStage.RENDER_SCENES,
    PipelineStage.FACEFIX,
    PipelineStage.UPSCALE,
})


def build_run_state(args: argparse.Namespace, stages: list[PipelineStage]) -> PipelineRunState:
    context = build_run_context(args)
    app_config_path = resolve_runner_path(args.app_config)
    single_prompt_workflow = args.single_prompt_workflow
    if single_prompt_workflow is None or (
        args.video_pipeline.startswith("minimax-h3-")
        and Path(str(single_prompt_workflow)).as_posix().casefold()
        == Path(default_single_prompt_workflow("ltx_i2v")).as_posix().casefold()
    ):
        single_prompt_workflow = default_single_prompt_workflow(args.video_pipeline)
    state = PipelineRunState(
        args=args,
        context=context,
        app_config_path=app_config_path,
        storyboard_workflow=resolve_runner_path(args.storyboard_workflow),
        reference_hero_workflow=resolve_runner_path(args.reference_hero_workflow),
        reference_edit_workflow=resolve_runner_path(args.reference_edit_workflow),
        msr_workflow=resolve_runner_path(args.msr_workflow),
        ingredients_workflow=resolve_runner_path(args.ingredients_workflow),
        relay_workflow=resolve_runner_path(args.relay_workflow) if str(args.relay_workflow).strip() else Path(),
        single_prompt_workflow=resolve_runner_path(single_prompt_workflow),
        facefix_workflow=resolve_runner_path(args.facefix_workflow),
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


def run(
    args: argparse.Namespace,
    *,
    on_stage_complete: Callable[[str], None] | None = None,
) -> PipelineRunResult:
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
        if on_stage_complete is not None:
            on_stage_complete(stage.value)

    console.print("Pipeline complete.")
    console.print(f"Render plan: {state.plan_for_next_step}")
    if state.final_video_path:
        console.print(f"Final video: {state.final_video_path}")
    elif state.video_only_path:
        console.print(f"Video-only concat: {state.video_only_path}")
    if state.openshot_project_path:
        console.print(f"OpenShot project: {state.openshot_project_path}")
    if getattr(state, "timeline_project_path", None):
        console.print(f"Timeline project: {state.timeline_project_path}")

    return PipelineRunResult(
        render_plan_path=state.plan_for_next_step,
        final_video_path=state.final_video_path,
        video_only_path=state.video_only_path,
        openshot_project_path=state.openshot_project_path,
        timeline_project_path=getattr(state, "timeline_project_path", None),
    )


def main() -> None:
    run(build_arg_parser().parse_args())
