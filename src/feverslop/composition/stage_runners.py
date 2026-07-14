from __future__ import annotations

from pathlib import Path
import argparse
import json
import subprocess

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from feverslop.adapters.openai_compatible_llm import OpenAICompatibleLLMClient
from feverslop.adapters.video_postprocessor import VideoPostProcessor
from feverslop.application.generate_render_plan import GenerateRenderPlanRequest
from feverslop.application.msr_prompt_enrichment import enrich_render_plan_with_msr_prompts
from feverslop.application.reference_bible import enrich_render_plan_with_reference_sheets
from feverslop.application.render_storyboard import RenderStoryboardRequest
from feverslop.application.render_video import RenderVideoScenesRequest
from feverslop.composition.generate_render_plan import build_generate_render_plan_use_case  # noqa: F401
from feverslop.composition.generate_render_plan import execute_generate_render_plan
from feverslop.composition.render_storyboard import build_render_storyboard_use_case
from feverslop.composition.render_video import RenderVideoCompositionOptions, build_render_video_scenes_use_case
from feverslop.config.app_config import AppConfig
from feverslop.ports.rendering import WorkflowAnchorConfig
from feverslop.prompting.ltx_prompt_anchor_fixer import LTXPromptAnchorFixer, validate_anchor_file
from feverslop.prompting.relay_direction_builder import RelayDirectionBuilder
from feverslop.tools.reference_bible import build_arg_parser as build_reference_bible_arg_parser
from feverslop.tools.reference_bible import run as render_reference_bible
from feverslop.tools.storyboard_page import parse_scene_list
from feverslop.tools.storyboard_page import generate_storyboard_page

from .arg_parser import PipelineStage
from .config_loader import PipelineRunContext, PipelineRunState, count_render_plan_items

_REFERENCE_BIBLE_PARSER = None


def _get_reference_bible_parser():
    global _REFERENCE_BIBLE_PARSER
    if _REFERENCE_BIBLE_PARSER is None:
        _REFERENCE_BIBLE_PARSER = build_reference_bible_arg_parser()
    return _REFERENCE_BIBLE_PARSER


console = Console()


class RenderProgressReporter:
    def __init__(self, description: str, total: int, *, console: Console = console):
        self.description = description
        self.total = total
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        self.task_id = None

    def __enter__(self) -> RenderProgressReporter:
        self.progress.__enter__()
        self.task_id = self.progress.add_task(self.description, total=self.total)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.progress.__exit__(exc_type, exc_value, traceback)

    def update(self, _output_path: Path, completed: int, _total: int) -> None:
        if self.task_id is not None:
            self.progress.update(self.task_id, completed=completed)


def _run_tests_stage(_state: PipelineRunState) -> None:
    run_unittest_suite()


def _run_main_pipeline_stage(state: PipelineRunState) -> None:
    execute_generate_render_plan(
        GenerateRenderPlanRequest(
            project_config_path=state.context.project_config_path,
            app_config_path=state.app_config_path,
            concept_batch_size=int(state.args.concept_batch_size),
        ),
        console=console,
    )
    state.plan_for_next_step = state.context.render_plan


def _run_relay_compact_stage(state: PipelineRunState) -> None:
    if state.args.render_mode == "single_prompt":
        raise ValueError("relay_compact requires render_mode relay or auto")
    app_config = AppConfig.load(state.app_config_path, required_keys=["llm", "comfyui"])
    llm = OpenAICompatibleLLMClient(
        base_url=app_config.llm.base_url,
        model=app_config.llm.model,
        temperature=app_config.llm.temperature,
        max_tokens=app_config.llm.max_tokens,
    )
    state.plan_for_next_step = RelayDirectionBuilder(llm=llm).compact_render_plan_file(
        input_render_plan=state.plan_for_next_step,
        output_render_plan=state.context.compact_plan,
    )


def _run_anchor_fix_stage(state: PipelineRunState) -> None:
    resolved_context = json.loads(state.context.resolved_context.read_text(encoding="utf-8-sig"))
    subject_anchor = str(resolved_context.get("subject", "")).strip()
    if not subject_anchor:
        raise ValueError(f"No subject anchor found in {state.context.resolved_context}")

    state.plan_for_next_step = LTXPromptAnchorFixer(subject_anchor=subject_anchor).fix_file(
        input_render_plan=state.plan_for_next_step,
        output_render_plan=state.context.anchored_plan,
    )
    warnings = validate_anchor_file(state.plan_for_next_step, subject_hint=subject_anchor)
    for warning in warnings[:30]:
        console.print(f"! {warning}")


def _run_storyboard_frames_stage(state: PipelineRunState) -> None:
    if state.args.video_pipeline == "ltx_msr":
        raise ValueError("storyboard_frames is not used by ltx_msr")
    app_config = AppConfig.load(state.app_config_path, required_keys=["llm", "comfyui"])
    storyboard_use_case = build_render_storyboard_use_case(
        app_config=app_config,
        workflow_path=state.storyboard_workflow,
        output_dir=state.context.storyboard_dir,
    )
    storyboard_total = count_render_plan_items(state.plan_for_next_step)
    with RenderProgressReporter("Rendering storyboard frames", storyboard_total) as storyboard_progress:
        storyboard_use_case.execute(
            RenderStoryboardRequest(
                render_plan_path=state.plan_for_next_step,
                workflow_path=state.storyboard_workflow,
                output_dir=state.context.storyboard_dir,
                character_lora_strength=state.args.storyboard_lora_strength,
                on_frame_complete=storyboard_progress.update,
            )
        )


def _run_storyboard_page_stage(state: PipelineRunState) -> None:
    if state.args.video_pipeline == "ltx_msr":
        raise ValueError("storyboard_page is not used by ltx_msr")
    generate_storyboard_page(
        render_plan_path=state.plan_for_next_step,
        storyboard_dir=state.context.storyboard_dir,
        output_html=state.context.storyboard_page,
    )


def _run_msr_references_stage(state: PipelineRunState) -> None:
    if state.args.video_pipeline not in ("ltx_msr", "ltx_ingredients"):
        raise ValueError("msr_references requires --video-pipeline ltx_msr or ltx_ingredients")
    reference_args = _get_reference_bible_parser().parse_args([
        "--project-config",
        str(state.context.project_config_path),
        "--app-config",
        str(state.app_config_path),
        "--hero-workflow",
        str(state.reference_hero_workflow),
        "--edit-workflow",
        str(state.reference_edit_workflow),
        "--output-dir",
        str(state.context.references_dir),
        "--view-set",
        "msr",
    ])
    render_reference_bible(reference_args)


def _run_msr_reference_sheets_stage(state: PipelineRunState) -> None:
    if state.args.video_pipeline not in ("ltx_msr", "ltx_ingredients"):
        raise ValueError("msr_reference_sheets requires --video-pipeline ltx_msr or ltx_ingredients")
    msr_reference_total = count_render_plan_items(state.plan_for_next_step)
    with RenderProgressReporter("Enriching MSR references", msr_reference_total) as reference_progress:
        state.plan_for_next_step = enrich_render_plan_with_reference_sheets(
            state.plan_for_next_step,
            state.context.references_dir,
            state.context.reference_plan,
            on_scene_complete=_scene_progress_callback(reference_progress),
        )


def _run_msr_prompt_enrich_stage(state: PipelineRunState) -> None:
    if state.args.video_pipeline not in ("ltx_msr", "ltx_ingredients"):
        raise ValueError("msr_prompt_enrich requires --video-pipeline ltx_msr or ltx_ingredients")
    app_config = AppConfig.load(state.app_config_path, required_keys=["llm", "comfyui"])
    llm = OpenAICompatibleLLMClient(
        base_url=app_config.llm.base_url,
        model=app_config.llm.model,
        temperature=app_config.llm.temperature,
        max_tokens=app_config.llm.max_tokens,
    )
    msr_prompt_total = count_render_plan_items(state.plan_for_next_step)
    with RenderProgressReporter("Enriching MSR prompts", msr_prompt_total) as msr_prompt_progress:
        state.plan_for_next_step = enrich_render_plan_with_msr_prompts(
            state.plan_for_next_step,
            state.context.reference_plan,
            llm=llm,
            on_scene_complete=_scene_progress_callback(msr_prompt_progress),
        )


def _run_ingredients_sheets_stage(state: PipelineRunState) -> None:
    from feverslop.application.render_plan_ingredients_sheets import enrich_render_plan_with_ingredients_sheets
    if state.args.video_pipeline != "ltx_ingredients":
        raise ValueError("ingredients_sheets requires --video-pipeline ltx_ingredients")
    ingredients_total = count_render_plan_items(state.plan_for_next_step)
    with RenderProgressReporter("Composing Ingredients scene sheets", ingredients_total) as progress:
        state.plan_for_next_step = enrich_render_plan_with_ingredients_sheets(
            state.plan_for_next_step,
            state.context.references_dir,
            state.context.ingredients_plan,
            on_scene_complete=_scene_progress_callback(progress),
        )


def _run_ltx_render_scenes_stage(state: PipelineRunState) -> None:
    if state.args.video_pipeline not in ("ltx_msr", "ltx_ingredients") and state.args.render_mode != "single_prompt" and not str(state.args.relay_workflow).strip():
        raise ValueError(f"RenderMode '{state.args.render_mode}' requires --relay-workflow pointing to a workflow with #PROMPT_RELAY.")

    if state.args.video_pipeline == "ltx_msr":
        ltx_workflow = state.msr_workflow
        ltx_single_prompt_workflow = None
    elif state.args.video_pipeline == "ltx_ingredients":
        ltx_workflow = state.ingredients_workflow
        ltx_single_prompt_workflow = None
    else:
        ltx_workflow = state.single_prompt_workflow if state.args.render_mode == "single_prompt" else state.relay_workflow
        ltx_single_prompt_workflow = state.single_prompt_workflow if state.args.render_mode == "auto" else None
    video_use_case = build_render_video_scenes_use_case(
        RenderVideoCompositionOptions(
            app_config_path=state.app_config_path,
            project_config_path=state.context.project_config_path,
            render_plan_path=state.plan_for_next_step,
            workflow_path=ltx_workflow,
            output_dir=state.context.ltx_dir,
            video_pipeline=state.args.video_pipeline,
            single_prompt_workflow_path=ltx_single_prompt_workflow,
            render_mode=state.args.render_mode,
            single_prompt_title=state.args.single_prompt_title,
            single_prompt_input=state.args.single_prompt_input,
            character_lora_strength=state.args.video_character_lora_strength,
            lora_1_strength_model=state.args.video_lora_1_strength_model,
            lora_1_strength_clip=state.args.video_lora_1_strength_clip,
            lora_split_enabled=state.args.lora_split_enabled,
            randomize_seed=state.args.randomize_seed,
            debug_workflows_dir=state.context.ltx_debug_dir,
            rolling_frame_profile=state.args.rolling_frame_profile,
        ),
        console=console,
    )
    ltx_scene_numbers = {state.args.smoke_scene} if state.args.smoke_only else parse_scene_list(state.args.scenes)
    ltx_total = count_render_plan_items(state.plan_for_next_step, scene_numbers=ltx_scene_numbers)
    with RenderProgressReporter("Rendering LTX scenes", ltx_total) as ltx_progress:
        video_use_case.execute(
            RenderVideoScenesRequest(
                render_plan_path=state.plan_for_next_step,
                workflow_path=ltx_workflow,
                audio_file=state.context.input_audio,
                storyboard_dir=state.context.storyboard_dir,
                output_dir=state.context.ltx_dir,
                render_mode=state.args.render_mode,
                single_prompt_workflow_path=ltx_single_prompt_workflow,
                scene_numbers=ltx_scene_numbers,
                skip_existing=False if state.args.smoke_only else not state.args.no_skip_existing,
                anchors=WorkflowAnchorConfig(
                    single_prompt_title=state.args.single_prompt_title,
                    single_prompt_input=state.args.single_prompt_input,
                ),
                on_scene_complete=ltx_progress.update,
            )
        )


def _run_concat_video_only_stage(state: PipelineRunState) -> None:
    from .config_loader import rewrite_concat_list, collect_render_plan_scene_clips
    rewrite_concat_list(collect_render_plan_scene_clips(state.plan_for_next_step, state.context.ltx_dir), state.context.ltx_dir)
    postprocessor = VideoPostProcessor(ffmpeg_path="ffmpeg", audio_bitrate="320k")
    state.video_only_path = postprocessor.concat_clips(
        concat_list=state.context.concat_list,
        output_file=state.context.final_concat_video,
        video_only=True,
    )


def _run_mux_original_audio_stage(state: PipelineRunState) -> None:
    video_only_path = state.video_only_path or state.context.final_concat_video
    if state.video_only_path is None and not Path(video_only_path).exists():
        raise FileNotFoundError(f"Video-only concat not found: {video_only_path}")
    postprocessor = VideoPostProcessor(ffmpeg_path="ffmpeg", audio_bitrate="320k")
    state.final_video_path = postprocessor.mux_original_audio(
        video_file=video_only_path,
        audio_file=state.context.input_audio,
        output_file=state.context.final_concat,
    )


def _run_diagnostic_scene_audio_concat_stage(state: PipelineRunState) -> None:
    postprocessor = VideoPostProcessor(ffmpeg_path="ffmpeg", audio_bitrate="320k")
    postprocessor.concat_clips(
        concat_list=state.context.concat_list,
        output_file=state.context.final_concat_scene_audio_debug,
        video_only=False,
    )


STAGE_RUNNERS = {
    PipelineStage.TESTS: _run_tests_stage,
    PipelineStage.MAIN_PIPELINE: _run_main_pipeline_stage,
    PipelineStage.RELAY_COMPACT: _run_relay_compact_stage,
    PipelineStage.ANCHOR_FIX: _run_anchor_fix_stage,
    PipelineStage.STORYBOARD_FRAMES: _run_storyboard_frames_stage,
    PipelineStage.STORYBOARD_PAGE: _run_storyboard_page_stage,
    PipelineStage.MSR_REFERENCES: _run_msr_references_stage,
    PipelineStage.MSR_REFERENCE_SHEETS: _run_msr_reference_sheets_stage,
    PipelineStage.MSR_PROMPT_ENRICH: _run_msr_prompt_enrich_stage,
    PipelineStage.INGREDIENTS_SHEETS: _run_ingredients_sheets_stage,
    PipelineStage.LTX_RENDER_SCENES: _run_ltx_render_scenes_stage,
    PipelineStage.CONCAT_VIDEO_ONLY: _run_concat_video_only_stage,
    PipelineStage.MUX_ORIGINAL_AUDIO: _run_mux_original_audio_stage,
    PipelineStage.DIAGNOSTIC_SCENE_AUDIO_CONCAT: _run_diagnostic_scene_audio_concat_stage,
}

STAGE_LABELS = {
    PipelineStage.TESTS: "tests",
    PipelineStage.MAIN_PIPELINE: "Main pipeline",
    PipelineStage.RELAY_COMPACT: "relay compact",
    PipelineStage.ANCHOR_FIX: "anchor fix",
    PipelineStage.STORYBOARD_FRAMES: "Storyboard frames",
    PipelineStage.STORYBOARD_PAGE: "Storyboard page",
    PipelineStage.MSR_REFERENCES: "MSR references",
    PipelineStage.MSR_REFERENCE_SHEETS: "MSR reference sheets",
    PipelineStage.MSR_PROMPT_ENRICH: "MSR prompt enrichment",
    PipelineStage.INGREDIENTS_SHEETS: "Ingredients scene sheets",
    PipelineStage.LTX_RENDER_SCENES: "LTX render",
    PipelineStage.CONCAT_VIDEO_ONLY: "Final concat video-only",
    PipelineStage.MUX_ORIGINAL_AUDIO: "Mux original audio",
    PipelineStage.DIAGNOSTIC_SCENE_AUDIO_CONCAT: "Diagnostic scene-audio concat",
}


def run_unittest_suite() -> None:
    subprocess.run(["uv", "run", "python", "-m", "unittest", "discover", "-s", "tests"], check=True)


def _scene_progress_callback(progress: RenderProgressReporter):
    def update(scene_number: int, completed: int, total: int) -> None:
        progress.update(Path(f"scene_{scene_number:04}.json"), completed, total)

    return update


def write_step(message: str) -> None:
    console.print()
    console.print(f"==> {message}")


def resolve_pipeline_stages(args: argparse.Namespace) -> list[PipelineStage]:
    selected = getattr(args, "stages", None)
    if selected:
        return [PipelineStage(stage) for stage in selected]

    stages: list[PipelineStage] = []
    if not args.skip_tests:
        stages.append(PipelineStage.TESTS)
    if not args.skip_main_pipeline:
        stages.append(PipelineStage.MAIN_PIPELINE)
    else:
        console.print("Skipping main pipeline; using existing timeline, prompts, and render plan.")
    if not args.skip_relay_compact and args.render_mode != "single_prompt":
        stages.append(PipelineStage.RELAY_COMPACT)
    if not args.skip_anchor_fix:
        stages.append(PipelineStage.ANCHOR_FIX)
    if args.video_pipeline == "ltx_msr":
        if not args.skip_msr_reference_render:
            stages.append(PipelineStage.MSR_REFERENCES)
        else:
            console.print("Skipping MSR reference rendering; using existing reference manifests.")
        stages.append(PipelineStage.MSR_REFERENCE_SHEETS)
        if not args.skip_msr_prompt_enrichment:
            stages.append(PipelineStage.MSR_PROMPT_ENRICH)
        else:
            console.print("Skipping MSR prompt enrichment; using existing MSR prompt fields.")
    elif args.video_pipeline == "ltx_ingredients":
        if not args.skip_msr_reference_render:
            stages.append(PipelineStage.MSR_REFERENCES)
        else:
            console.print("Skipping MSR reference rendering; using existing reference manifests.")
        stages.append(PipelineStage.MSR_REFERENCE_SHEETS)
        if not args.skip_msr_prompt_enrichment:
            stages.append(PipelineStage.MSR_PROMPT_ENRICH)
        else:
            console.print("Skipping MSR prompt enrichment; using existing MSR prompt fields.")
        if not getattr(args, "skip_ingredients_sheets", False):
            stages.append(PipelineStage.INGREDIENTS_SHEETS)
        else:
            console.print("Skipping Ingredients sheets; using existing sheets or references.")
    else:
        if not args.skip_storyboard:
            stages.append(PipelineStage.STORYBOARD_FRAMES)
        if not args.skip_storyboard_page:
            stages.append(PipelineStage.STORYBOARD_PAGE)
    if not args.skip_ltx:
        stages.append(PipelineStage.LTX_RENDER_SCENES)
    if not args.skip_final_concat:
        stages.append(PipelineStage.CONCAT_VIDEO_ONLY)
        stages.append(PipelineStage.MUX_ORIGINAL_AUDIO)
        if args.diagnostic_original_audio_mux:
            stages.append(PipelineStage.DIAGNOSTIC_SCENE_AUDIO_CONCAT)
        elif args.no_original_audio_mux:
            console.print("--no-original-audio-mux is deprecated; original-audio muxing is now always used for final concat.")
    return stages


def _initial_render_plan(context: PipelineRunContext, args: argparse.Namespace, stages: list[PipelineStage]) -> Path:
    upstream_stages = {PipelineStage.MAIN_PIPELINE, PipelineStage.RELAY_COMPACT, PipelineStage.ANCHOR_FIX, PipelineStage.MSR_REFERENCE_SHEETS, PipelineStage.INGREDIENTS_SHEETS}
    if args.video_pipeline == "ltx_msr" and context.reference_plan.exists() and not upstream_stages.intersection(stages):
        return context.reference_plan
    if args.video_pipeline == "ltx_ingredients" and context.ingredients_plan.exists() and not upstream_stages.intersection(stages):
        return context.ingredients_plan
    return context.render_plan
