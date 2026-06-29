from __future__ import annotations

from dataclasses import dataclass
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FeverSlop pipeline from Python.")
    parser.add_argument("project_root", nargs="?", default=None)
    parser.add_argument("--project-config", default=None)
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
    context = build_run_context(args)
    app_config_path = resolve_runner_path(args.app_config)
    storyboard_workflow = resolve_runner_path(args.storyboard_workflow)
    reference_hero_workflow = resolve_runner_path(args.reference_hero_workflow)
    reference_edit_workflow = resolve_runner_path(args.reference_edit_workflow)
    msr_workflow = resolve_runner_path(args.msr_workflow)
    relay_workflow = resolve_runner_path(args.relay_workflow) if str(args.relay_workflow).strip() else Path("")
    single_prompt_workflow = resolve_runner_path(args.single_prompt_workflow)
    console.print(f"Project: {context.project_config_path}")
    console.print(f"Input audio: {context.input_audio}")
    console.print(f"Song ID: {context.song_id}")
    console.print(f"Render mode: {args.render_mode}")

    if not args.skip_tests:
        write_step("Running tests")
        run_unittest_suite()

    if not args.skip_main_pipeline:
        write_step("Running main pipeline")
        build_generate_render_plan_use_case(console=console).execute(
            GenerateRenderPlanRequest(
                project_config_path=context.project_config_path,
                app_config_path=app_config_path,
                concept_batch_size=int(args.concept_batch_size),
            )
        )
    else:
        console.print("Skipping main pipeline; using existing timeline, prompts, and render plan.")

    plan_for_next_step = context.render_plan
    if not args.skip_relay_compact and args.render_mode != "single_prompt":
        write_step("Compacting relay prompts")
        app_config = AppConfig.load(app_config_path)
        llm = OpenAICompatibleLLMClient(
            base_url=app_config.llm.base_url,
            model=app_config.llm.model,
            temperature=app_config.llm.temperature,
            max_tokens=app_config.llm.max_tokens,
        )
        plan_for_next_step = RelayDirectionBuilder(llm=llm).compact_render_plan_file(
            input_render_plan=plan_for_next_step,
            output_render_plan=context.compact_plan,
        )

    if not args.skip_anchor_fix:
        resolved_context = json.loads(context.resolved_context.read_text(encoding="utf-8-sig"))
        subject_anchor = str(resolved_context.get("subject", "")).strip()
        if not subject_anchor:
            raise ValueError(f"No subject anchor found in {context.resolved_context}")

        write_step("Fixing prompt anchors")
        plan_for_next_step = LTXPromptAnchorFixer(subject_anchor=subject_anchor).fix_file(
            input_render_plan=plan_for_next_step,
            output_render_plan=context.anchored_plan,
        )
        warnings = validate_anchor_file(plan_for_next_step, subject_hint=subject_anchor)
        for warning in warnings[:30]:
            console.print(f"! {warning}")

    should_render_storyboard = not args.skip_storyboard and args.video_pipeline != "ltx_msr"
    if should_render_storyboard:
        write_step("Rendering storyboard")
        app_config = AppConfig.load(app_config_path)
        storyboard_use_case = build_render_storyboard_use_case(
            app_config=app_config,
            workflow_path=storyboard_workflow,
            output_dir=context.storyboard_dir,
        )
        storyboard_total = count_render_plan_items(plan_for_next_step)
        with RenderProgressReporter("Rendering storyboard frames", storyboard_total) as storyboard_progress:
            storyboard_use_case.execute(
                RenderStoryboardRequest(
                    render_plan_path=plan_for_next_step,
                    workflow_path=storyboard_workflow,
                    output_dir=context.storyboard_dir,
                    character_lora_strength=args.storyboard_lora_strength,
                    on_frame_complete=storyboard_progress.update,
                )
            )

    should_render_storyboard_page = not args.skip_storyboard_page and args.video_pipeline != "ltx_msr"
    if should_render_storyboard_page:
        write_step("Generating storyboard page")
        generate_storyboard_page(
            render_plan_path=plan_for_next_step,
            storyboard_dir=context.storyboard_dir,
            output_html=context.storyboard_page,
        )

    if args.video_pipeline == "ltx_msr":
        if not args.skip_msr_reference_render:
            write_step("Rendering MSR references")
            reference_args = build_reference_bible_arg_parser().parse_args([
                "--project-config",
                str(context.project_config_path),
                "--app-config",
                str(app_config_path),
                "--hero-workflow",
                str(reference_hero_workflow),
                "--edit-workflow",
                str(reference_edit_workflow),
                "--output-dir",
                str(context.references_dir),
                "--view-set",
                "msr",
            ])
            render_reference_bible(reference_args)
        else:
            console.print("Skipping MSR reference rendering; using existing reference manifests.")

        write_step("Enriching render plan with MSR references")
        plan_for_next_step = enrich_render_plan_with_reference_sheets(
            plan_for_next_step,
            context.references_dir,
            context.reference_plan,
        )

    if not args.skip_ltx:
        write_step("Rendering LTX")
        if args.video_pipeline != "ltx_msr" and args.render_mode != "single_prompt" and not str(args.relay_workflow).strip():
            raise ValueError(f"RenderMode '{args.render_mode}' requires --relay-workflow pointing to a workflow with #PROMPT_RELAY.")

        if args.video_pipeline == "ltx_msr":
            ltx_workflow = msr_workflow
            ltx_single_prompt_workflow = None
        else:
            ltx_workflow = single_prompt_workflow if args.render_mode == "single_prompt" else relay_workflow
            ltx_single_prompt_workflow = single_prompt_workflow if args.render_mode == "auto" else None
        video_use_case = build_render_video_scenes_use_case(
            RenderVideoCompositionOptions(
                app_config_path=app_config_path,
                project_config_path=context.project_config_path,
                render_plan_path=plan_for_next_step,
                workflow_path=ltx_workflow,
                output_dir=context.ltx_dir,
                video_pipeline=args.video_pipeline,
                single_prompt_workflow_path=ltx_single_prompt_workflow,
                render_mode=args.render_mode,
                single_prompt_title=args.single_prompt_title,
                single_prompt_input=args.single_prompt_input,
                character_lora_strength=args.video_character_lora_strength,
                lora_1_strength_model=args.video_lora_1_strength_model,
                lora_1_strength_clip=args.video_lora_1_strength_clip,
                lora_split_enabled=args.lora_split_enabled,
                randomize_seed=args.randomize_seed,
                debug_workflows_dir=context.ltx_debug_dir,
                rolling_frame_profile=args.rolling_frame_profile,
            ),
            console=console,
        )
        ltx_scene_numbers = {args.smoke_scene} if args.smoke_only else None
        ltx_total = count_render_plan_items(plan_for_next_step, scene_numbers=ltx_scene_numbers)
        with RenderProgressReporter("Rendering LTX scenes", ltx_total) as ltx_progress:
            rendered_ltx_clips = video_use_case.execute(
                RenderVideoScenesRequest(
                    render_plan_path=plan_for_next_step,
                    workflow_path=ltx_workflow,
                    audio_file=context.input_audio,
                    storyboard_dir=context.storyboard_dir,
                    output_dir=context.ltx_dir,
                    render_mode=args.render_mode,
                    single_prompt_workflow_path=ltx_single_prompt_workflow,
                    scene_numbers=ltx_scene_numbers,
                    skip_existing=False if args.smoke_only else not args.no_skip_existing,
                    anchors=WorkflowAnchorConfig(
                        single_prompt_title=args.single_prompt_title,
                        single_prompt_input=args.single_prompt_input,
                    ),
                    on_scene_complete=ltx_progress.update,
                )
            )
        rewrite_concat_list(rendered_ltx_clips, context.ltx_dir)

    video_only_path = None
    final_video_path = None
    if not args.skip_final_concat and context.concat_list.exists():
        postprocessor = VideoPostProcessor(ffmpeg_path="ffmpeg", audio_bitrate="320k")
        write_step("Final FFmpeg video-only concat")
        video_only_path = postprocessor.concat_clips(
            concat_list=context.concat_list,
            output_file=context.final_concat_video,
            video_only=True,
        )
        write_step("Muxing original full audio")
        final_video_path = postprocessor.mux_original_audio(
            video_file=video_only_path,
            audio_file=context.input_audio,
            output_file=context.final_concat,
        )
        if args.diagnostic_original_audio_mux:
            write_step("Diagnostic concat with per-scene audio")
            postprocessor.concat_clips(
                concat_list=context.concat_list,
                output_file=context.final_concat_scene_audio_debug,
                video_only=False,
            )
        elif args.no_original_audio_mux:
            console.print("--no-original-audio-mux is deprecated; original-audio muxing is now always used for final concat.")

    console.print("Pipeline complete.")
    console.print(f"Render plan: {plan_for_next_step}")
    if final_video_path:
        console.print(f"Final video: {final_video_path}")
    elif video_only_path:
        console.print(f"Video-only concat: {video_only_path}")

    return PipelineRunResult(
        render_plan_path=plan_for_next_step,
        final_video_path=final_video_path,
        video_only_path=video_only_path,
    )


def run_unittest_suite() -> None:
    subprocess.run(["uv", "run", "python", "-m", "unittest", "discover", "-s", "tests"], check=True)


def write_step(message: str) -> None:
    console.print()
    console.print(f"==> {message}")


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
