from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import argparse
import json
import os
import re
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
from feverslop.adapters.pipeline_runner_options import add_runner_options
from feverslop.adapters.video_postprocessor import VideoPostProcessor
from feverslop.application.generate_render_plan import GenerateRenderPlanRequest
from feverslop.application.msr_prompt_enrichment import enrich_render_plan_with_msr_prompts
from feverslop.application.reference_bible import enrich_render_plan_with_reference_sheets
from feverslop.application.render_storyboard import RenderStoryboardRequest
from feverslop.application.render_video import RenderVideoScenesRequest
from feverslop.composition.generate_render_plan import build_generate_render_plan_use_case
from feverslop.composition.render_storyboard import build_render_storyboard_use_case
from feverslop.composition.render_video import RenderVideoCompositionOptions, build_render_video_scenes_use_case
from feverslop.config.app_config import AppConfig
from feverslop.path_utils import coerce_local_path
from feverslop.ports.rendering import WorkflowAnchorConfig
from feverslop.prompting.ltx_prompt_anchor_fixer import LTXPromptAnchorFixer, validate_anchor_file
from feverslop.prompting.relay_direction_builder import RelayDirectionBuilder
from feverslop.tools.reference_bible import build_arg_parser as build_reference_bible_arg_parser
from feverslop.tools.reference_bible import run as render_reference_bible
from feverslop.tools.storyboard_page import parse_scene_list
from feverslop.tools.storyboard_page import generate_storyboard_page


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


@dataclass(frozen=True)
class PipelineRunContext:
    project_config_path: Path
    project_config_dir: Path
    input_audio: Path
    song_id: str
    project_file_stem: str
    project_output_dir: Path
    timeline_dir: Path
    prompts_dir: Path
    render_dir: Path
    stage1_segments: Path
    resolved_context: Path
    concept_prompts: Path
    scene_details: Path
    scene_prompts: Path
    render_plan: Path
    reference_plan: Path
    references_dir: Path
    compact_plan: Path
    anchored_plan: Path
    storyboard_dir: Path
    storyboard_page: Path
    ltx_dir: Path
    ltx_debug_dir: Path
    final_concat_video: Path
    final_concat: Path
    final_concat_scene_audio_debug: Path
    concat_list: Path


@dataclass(frozen=True)
class PipelineRunResult:
    render_plan_path: Path
    final_video_path: Path | None = None
    video_only_path: Path | None = None


class PipelineStage(str, Enum):
    TESTS = "tests"
    MAIN_PIPELINE = "main_pipeline"
    RELAY_COMPACT = "relay_compact"
    ANCHOR_FIX = "anchor_fix"
    STORYBOARD_FRAMES = "storyboard_frames"
    STORYBOARD_PAGE = "storyboard_page"
    MSR_REFERENCES = "msr_references"
    MSR_REFERENCE_SHEETS = "msr_reference_sheets"
    MSR_PROMPT_ENRICH = "msr_prompt_enrich"
    LTX_RENDER_SCENES = "ltx_render_scenes"
    CONCAT_VIDEO_ONLY = "concat_video_only"
    MUX_ORIGINAL_AUDIO = "mux_original_audio"
    DIAGNOSTIC_SCENE_AUDIO_CONCAT = "diagnostic_scene_audio_concat"


@dataclass
class PipelineRunState:
    args: argparse.Namespace
    context: PipelineRunContext
    app_config_path: Path
    storyboard_workflow: Path
    reference_hero_workflow: Path
    reference_edit_workflow: Path
    msr_workflow: Path
    relay_workflow: Path
    single_prompt_workflow: Path
    plan_for_next_step: Path
    video_only_path: Path | None = None
    final_video_path: Path | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FeverSlop pipeline from Python.")
    parser.add_argument("project_root", nargs="?", default=None)
    parser.add_argument("--project-config", default=None)
    parser.add_argument(
        "--stage",
        dest="stages",
        action="append",
        choices=[stage.value for stage in PipelineStage],
        default=None,
        help="Run only a specific atomic pipeline stage. May be passed more than once.",
    )
    add_runner_options(parser)
    return parser


def build_run_context(args: argparse.Namespace) -> PipelineRunContext:
    project_config = args.project_config
    project_root = args.project_root

    if not project_config:
        if not project_root:
            project_root = os.fspath(Path("projects") / "my_first_project")

        project_root_path = resolve_runner_path(project_root)
        if project_root_path.is_file():
            project_config_path = project_root_path
        else:
            project_config_path = project_root_path / "config.json"
    else:
        project_config_path = resolve_runner_path(project_config)

    project_config_path = project_config_path.resolve()
    project_config_dir = project_config_path.parent
    project_config_json = json.loads(project_config_path.read_text(encoding="utf-8-sig"))
    input_audio = coerce_local_path(str(project_config_json["input_audio"]), base_dir=project_config_dir)
    input_audio = input_audio.resolve()

    song_id = input_audio.stem
    project_file_stem = convert_to_safe_file_stem(project_config_json.get("project_name"), song_id)
    project_output_dir = project_config_dir / "output"
    timeline_dir = project_output_dir / "timeline"
    prompts_dir = project_output_dir / "prompts"
    render_dir = project_output_dir / "render"
    storyboard_dir = render_dir / "storyboard"
    ltx_dir = render_dir / ("ltx_msr" if getattr(args, "video_pipeline", "ltx_i2v") == "ltx_msr" else f"ltx_{args.render_mode}")
    if args.smoke_only:
        ltx_dir = render_dir / ("ltx_msr_smoke" if getattr(args, "video_pipeline", "ltx_i2v") == "ltx_msr" else f"ltx_{args.render_mode}_smoke")
    ltx_debug_dir = render_dir / ("ltx_msr_debug" if getattr(args, "video_pipeline", "ltx_i2v") == "ltx_msr" else f"ltx_{args.render_mode}_debug")

    return PipelineRunContext(
        project_config_path=project_config_path,
        project_config_dir=project_config_dir,
        input_audio=input_audio,
        song_id=song_id,
        project_file_stem=project_file_stem,
        project_output_dir=project_output_dir,
        timeline_dir=timeline_dir,
        prompts_dir=prompts_dir,
        render_dir=render_dir,
        stage1_segments=timeline_dir / f"stage1_segments_{song_id}.json",
        resolved_context=prompts_dir / f"resolved_context_{song_id}.json",
        concept_prompts=prompts_dir / f"concept_prompts_{song_id}.json",
        scene_details=prompts_dir / f"scene_details_{song_id}.json",
        scene_prompts=prompts_dir / f"scene_prompts_{song_id}.json",
        render_plan=render_dir / f"render_plan_{song_id}.json",
        reference_plan=render_dir / f"render_plan_{song_id}_refs.json",
        references_dir=project_output_dir / "references",
        compact_plan=render_dir / f"render_plan_{song_id}__compact.json",
        anchored_plan=render_dir / f"render_plan_{song_id}__compact_anchored.json",
        storyboard_dir=storyboard_dir,
        storyboard_page=storyboard_dir / "index.html",
        ltx_dir=ltx_dir,
        ltx_debug_dir=ltx_debug_dir,
        final_concat_video=ltx_dir / f"{project_file_stem}_video_only.mp4",
        final_concat=ltx_dir / f"{project_file_stem}.mp4",
        final_concat_scene_audio_debug=ltx_dir / f"{project_file_stem}_scene_audio_debug.mp4",
        concat_list=ltx_dir / "concat_list.txt",
    )


def convert_to_safe_file_stem(value, fallback: str) -> str:
    raw = str(value or "").strip() or fallback
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    safe = safe.strip("._-")
    return safe or fallback


def rewrite_concat_list(rendered_files: list[Path], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    concat_file = output_dir / "concat_list.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    with concat_file.open("w", encoding="utf-8") as f:
        for path in rendered_files:
            f.write(f"file '{Path(path).resolve().as_posix()}'\n")
    return concat_file


def collect_render_plan_scene_clips(render_plan_path: str | Path, output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    render_plan = json.loads(Path(render_plan_path).read_text(encoding="utf-8-sig"))
    clips: list[Path] = []
    missing: list[Path] = []
    for scene in render_plan:
        scene_number = int(scene["scene"])
        candidates = [
            output_dir / f"scene_{scene_number:04}.mp4",
            output_dir / "final" / f"scene_{scene_number:04}.mp4",
        ]
        clip = next((candidate for candidate in candidates if candidate.exists()), None)
        if clip is None:
            missing.append(candidates[0])
            continue
        clips.append(clip)

    if missing:
        raise FileNotFoundError(
            "Cannot build final concat; missing rendered scene clips: "
            + ", ".join(str(path) for path in missing[:10])
        )
    return clips


def count_render_plan_items(render_plan_path: str | Path, scene_numbers: set[int] | None = None, limit: int | None = None) -> int:
    render_plan = json.loads(Path(render_plan_path).read_text(encoding="utf-8-sig"))
    if scene_numbers is not None:
        render_plan = [scene for scene in render_plan if int(scene["scene"]) in scene_numbers]
    if limit is not None:
        render_plan = render_plan[:limit]
    return len(render_plan)


def runner_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_runner_path(value: str | Path) -> Path:
    return coerce_local_path(value, base_dir=runner_root())


def run(args: argparse.Namespace) -> PipelineRunResult:
    stages = resolve_pipeline_stages(args)
    state = build_run_state(args, stages)
    for stage in stages:
        write_step(f"Stage {STAGE_LABELS[stage]}")
        try:
            STAGE_RUNNERS[stage](state)
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
        relay_workflow=resolve_runner_path(args.relay_workflow) if str(args.relay_workflow).strip() else Path(""),
        single_prompt_workflow=resolve_runner_path(args.single_prompt_workflow),
        plan_for_next_step=_initial_render_plan(context, args, stages),
    )
    console.print(f"Project: {context.project_config_path}")
    console.print(f"Input audio: {context.input_audio}")
    console.print(f"Song ID: {context.song_id}")
    console.print(f"Render mode: {args.render_mode}")
    console.print("Stages: " + ", ".join(stage.value for stage in stages))
    return state


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
    upstream_stages = {PipelineStage.MAIN_PIPELINE, PipelineStage.RELAY_COMPACT, PipelineStage.ANCHOR_FIX, PipelineStage.MSR_REFERENCE_SHEETS}
    if args.video_pipeline == "ltx_msr" and context.reference_plan.exists() and not upstream_stages.intersection(stages):
        return context.reference_plan
    return context.render_plan


def _run_tests_stage(_state: PipelineRunState) -> None:
    run_unittest_suite()


def _run_main_pipeline_stage(state: PipelineRunState) -> None:
    build_generate_render_plan_use_case(console=console).execute(
        GenerateRenderPlanRequest(
            project_config_path=state.context.project_config_path,
            app_config_path=state.app_config_path,
            concept_batch_size=int(state.args.concept_batch_size),
        )
    )
    state.plan_for_next_step = state.context.render_plan


def _run_relay_compact_stage(state: PipelineRunState) -> None:
    if state.args.render_mode == "single_prompt":
        raise ValueError("relay_compact requires render_mode relay or auto")
    app_config = AppConfig.load(state.app_config_path)
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
    app_config = AppConfig.load(state.app_config_path)
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
    if state.args.video_pipeline != "ltx_msr":
        raise ValueError("msr_references requires --video-pipeline ltx_msr")
    reference_args = build_reference_bible_arg_parser().parse_args([
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
    if state.args.video_pipeline != "ltx_msr":
        raise ValueError("msr_reference_sheets requires --video-pipeline ltx_msr")
    msr_reference_total = count_render_plan_items(state.plan_for_next_step)
    with RenderProgressReporter("Enriching MSR references", msr_reference_total) as reference_progress:
        state.plan_for_next_step = enrich_render_plan_with_reference_sheets(
            state.plan_for_next_step,
            state.context.references_dir,
            state.context.reference_plan,
            on_scene_complete=_scene_progress_callback(reference_progress),
        )


def _run_msr_prompt_enrich_stage(state: PipelineRunState) -> None:
    if state.args.video_pipeline != "ltx_msr":
        raise ValueError("msr_prompt_enrich requires --video-pipeline ltx_msr")
    app_config = AppConfig.load(state.app_config_path)
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


def _run_ltx_render_scenes_stage(state: PipelineRunState) -> None:
    if state.args.video_pipeline != "ltx_msr" and state.args.render_mode != "single_prompt" and not str(state.args.relay_workflow).strip():
        raise ValueError(f"RenderMode '{state.args.render_mode}' requires --relay-workflow pointing to a workflow with #PROMPT_RELAY.")

    if state.args.video_pipeline == "ltx_msr":
        ltx_workflow = state.msr_workflow
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


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
