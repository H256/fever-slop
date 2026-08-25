from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from rich.console import Console
from rich.table import Table

from feverslop.adapters.pipeline_runner_options import RUNNER_ARGUMENTS
from feverslop.composition.resume_plan import build_compatibility_plan, build_resume_plan
from feverslop.composition.arg_parser import PUBLIC_PIPELINE_STAGES
from feverslop.composition.config_loader import resolve_runner_path
from feverslop.composition.pipeline_runner import run as pipeline_run
from feverslop.composition.project_render_settings import (
    resolve_project_render_settings,
)
from feverslop.composition.stage_runners import resolve_pipeline_stages
from feverslop.config.app_config import AppConfig, VramHandoffMode
from feverslop.domain.execution_plan import ExecutionPlan, PlanAction
from feverslop.domain.resource_phase import ResourcePhase, StageResource, select_first_resource_phase
from feverslop.errors import FeverSlopError
from feverslop.tools.storyboard_page import parse_scene_list


def build_run_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("run", help="Plan or safely resume a music-video pipeline.")
    parser.add_argument("project")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Show the immutable execution plan without writing.")
    mode.add_argument("--resume", action="store_true", help="Execute the minimal safe plan.")
    parser.add_argument(
        "--stage",
        dest="stages",
        action="append",
        choices=[stage.value for stage in PUBLIC_PIPELINE_STAGES],
        default=None,
        help="Advanced compatibility: select an atomic stage.",
    )
    for _name, flags, kwargs in RUNNER_ARGUMENTS:
        option = dict(kwargs)
        option["default"] = None
        parser.add_argument(*flags, **option)
    parser.add_argument(
        "--format",
        dest="timeline_format",
        choices=["mlt", "openshot", "both"],
        default=None,
        help="Advanced compatibility: timeline export format.",
    )
    return parser


def run_project_command(args: argparse.Namespace, *, console: Console | None = None) -> int:
    output = console or Console()
    project = Path(args.project).resolve()
    try:
        args.project_root = str(project)
        args.project_config = str(project / "config.json")
        requested_app_config = args.app_config
        requested_video_pipeline = args.video_pipeline
        args.app_config = args.app_config or _runner_default("app_config")
        resume_command = _resume_command(
            project,
            app_config=requested_app_config,
            scenes=args.scenes,
            video_pipeline=requested_video_pipeline,
        )
        explicit_runner_options = {
            name
            for name, _flags, _kwargs in RUNNER_ARGUMENTS
            if getattr(args, name, None) is not None
        }
        compatibility = _uses_compatibility_inputs(args)
        app_config = AppConfig.load(resolve_runner_path(args.app_config))
        args.video_pipeline = args.video_pipeline or _configured_pipeline(project)
        resolved = resolve_project_render_settings(
            project,
            video_pipeline=args.video_pipeline,
            explicit_runner_options=explicit_runner_options,
            reference_generation=args.reference_generation,
            sequence_to_sheet_workflow=args.sequence_to_sheet_workflow,
        )
        render_settings = resolved.settings if not compatibility else None
        args.project_render_settings = render_settings
        for name, value in resolved.runner_overrides.items():
            if getattr(args, name, None) is None:
                setattr(args, name, value)
        selected = parse_scene_list(args.scenes)
        if compatibility:
            _apply_runner_defaults(args)
            stages = resolve_pipeline_stages(args)
            plan = build_compatibility_plan(
                project,
                (stage.value for stage in stages),
                selected_scenes=selected,
            )
        else:
            plan = build_resume_plan(
                project,
                video_pipeline=args.video_pipeline,
                selected_scenes=selected,
                render_settings=render_settings,
            )
        _render_plan(plan, output)
        if plan.blocked:
            return 2
        manual_phase = _manual_phase(plan, app_config, compatibility=compatibility)
        if manual_phase is not None and manual_phase.stages:
            _render_manual_phase_preview(manual_phase, output)
        if args.dry_run:
            output.print("[dim]Dry run: no project artifacts were changed.[/dim]")
            return 0
        if not plan.runnable_stages:
            output.print("[green]Nothing to do; all selected artifacts are reusable.[/green]")
            return 0

        last_completed = "none"

        def remember(stage: str) -> None:
            nonlocal last_completed
            last_completed = stage

        try:
            if plan.mode == "compatibility":
                units = ((plan.runnable_stages, plan.runnable_scenes),)
            else:
                runnable_stages = (
                    manual_phase.stages
                    if manual_phase is not None
                    else plan.runnable_stages
                )
                units = tuple(
                    ((stage,), plan.runnable_scenes_for_stage(stage))
                    for stage in runnable_stages
                )
            for stages, scenes in units:
                run_args = argparse.Namespace(**vars(args))
                run_args.stages = list(stages)
                run_args.scenes = ",".join(str(scene) for scene in scenes) or None
                _apply_runner_defaults(run_args)
                pipeline_run(run_args, on_stage_complete=remember)
        except Exception as exc:
            output.print(f"[red]Run failed:[/red] {exc}")
            if _is_llm_loading_failure(exc):
                output.print(
                    "[yellow]LLM ist noch nicht bereit; Modell laden lassen oder den LLM-Dienst starten "
                    "und danach denselben Resume-Befehl erneut ausführen.[/yellow]",
                )
            output.print(f"Last completed stage: {last_completed}")
            output.print(f"Safe resume: {resume_command}")
            return 1
        if manual_phase is not None:
            next_resource = None
            if not compatibility:
                # Always replan after a manual phase.  The phase preview is
                # calculated before execution and can describe a resource
                # boundary that disappears once the phase materializes its
                # artifacts.
                next_plan = build_resume_plan(
                    project,
                    video_pipeline=args.video_pipeline,
                    selected_scenes=selected,
                    render_settings=render_settings,
                )
                next_phase = _manual_phase(next_plan, app_config, compatibility=False)
                if next_phase is not None and next_phase.stages:
                    next_resource = next_phase.resource
            if next_resource is not None:
                _render_manual_handoff(
                    manual_phase,
                    output,
                    resume_command,
                    target_resource=next_resource,
                )
        return 0
    except (FeverSlopError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        output.print(f"[red]Invalid/corrupt project:[/red] {exc}")
        output.print(f"Safe inspection: uv run python main.py status {project}")
        return 1


def _configured_pipeline(project: Path) -> str:
    payload = json.loads((project / "config.json").read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Project config must be an object")
    return str(payload.get("video_pipeline") or "ltx_i2v")


def _uses_compatibility_inputs(args: argparse.Namespace) -> bool:
    if args.stages:
        return True
    normal_options = {"app_config", "scenes", "video_pipeline"}
    return any(
        name not in normal_options and getattr(args, name, None) is not None
        for name, _flags, _kwargs in RUNNER_ARGUMENTS
    ) or getattr(args, "timeline_format", None) is not None


def _apply_runner_defaults(args: argparse.Namespace) -> None:
    for name, _flags, kwargs in RUNNER_ARGUMENTS:
        if getattr(args, name, None) is None and "default" in kwargs:
            setattr(args, name, kwargs["default"])
    if getattr(args, "timeline_format", None) is None:
        args.timeline_format = "both"


def _runner_default(name: str):
    return next(
        kwargs.get("default")
        for option_name, _flags, kwargs in RUNNER_ARGUMENTS
        if option_name == name
    )


def _manual_phase(
    plan: ExecutionPlan,
    app_config: AppConfig,
    *,
    compatibility: bool,
) -> ResourcePhase | None:
    if compatibility or app_config.execution.vram_handoff is not VramHandoffMode.MANUAL:
        return None
    return select_first_resource_phase(plan.runnable_stages)


def _render_manual_phase_preview(phase: ResourcePhase, output: Console) -> None:
    resource = phase.resource.value if phase.resource is not None else "CPU-only"
    output.print(f"[cyan]Next manual execution phase: {resource}[/cyan]")
    output.print("[dim]Stages: " + ", ".join(phase.stages) + "[/dim]")
    if phase.next_resource is not None:
        output.print(
            f"[yellow]Next required resource after that: {phase.next_resource.value}[/yellow]",
        )


def _is_llm_loading_failure(error: BaseException) -> bool:
    message = str(error).lower()
    return (
        "model is still loading" in message
        or "serviceunavailableerror" in message
        or "error code: 503" in message
    )


def _render_manual_handoff(
    phase: ResourcePhase,
    output: Console,
    resume_command: str,
    *,
    target_resource: StageResource | None = None,
) -> None:
    next_resource = target_resource if target_resource is not None else phase.next_resource
    if next_resource is StageResource.LLM:
        action = "unload ComfyUI and load the LLM"
    else:
        action = "unload the LLM and load ComfyUI"
    output.print(f"[yellow]Manual VRAM handoff required: {action}.[/yellow]")
    output.print(f"Then rerun: {resume_command}")


def _render_plan(plan: ExecutionPlan, output: Console) -> None:
    columns = ("Phase", "Scene", "Action", "Reason", "Stage") if plan.mode == "compatibility" else ("Phase", "Scene", "Action", "Reason")
    table = Table(*columns)
    for item in plan.items:
        values = [item.phase, str(item.scene) if item.scene is not None else "all", item.action.value, item.reason]
        if plan.mode == "compatibility":
            values.append(item.stage or "—")
        table.add_row(*values)
    output.print(table)
    if any(item.action is PlanAction.BLOCKED for item in plan.items):
        output.print("[red]Execution blocked; no pipeline stage was started.[/red]")


def _resume_command(
    project: Path,
    *,
    app_config: str | None = None,
    scenes: str | None = None,
    video_pipeline: str | None = None,
) -> str:
    command = ["uv", "run", "python", "main.py", "run", str(project), "--resume"]
    if app_config:
        command.extend(["--app-config", app_config])
    if scenes:
        command.extend(["--scenes", scenes])
    if video_pipeline:
        command.extend(["--video-pipeline", video_pipeline])
    return subprocess.list2cmdline(command)
