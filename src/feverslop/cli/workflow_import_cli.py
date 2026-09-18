"""CLI commands for per-project ComfyUI workflow import (P5).

Operates on ``projects/<slug>/workflows/`` via :class:`WorkflowImportStore`.
The test-run gate is enforced by the store: ``activate`` only succeeds when
validation AND a successful test-run are both recorded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from feverslop.ports.reporting import ReporterConsole
from feverslop.domain.workflow_import import TestRunResult
from feverslop.domain.workflow_import_inspector import inspect_workflow, validate_workflow
from feverslop.domain.workflow_import_families import spec_for
from feverslop.domain.workflow_import_store import WorkflowImportError, WorkflowImportStore


def build_workflow_import_parser(subparsers) -> None:
    """Register the per-project workflow import commands."""
    import_parser = subparsers.add_parser(
        "workflow-import",
        help="Import, validate, and activate per-project ComfyUI workflow snapshots.",
    )
    commands = import_parser.add_subparsers(dest="workflow_import_command", required=True)

    import_cmd = commands.add_parser(
        "import", help="Import a workflow file as a draft snapshot."
    )
    import_cmd.add_argument("--project-dir", required=True, help="Project directory.")
    import_cmd.add_argument("--profile-id", required=True, help="Local profile id.")
    import_cmd.add_argument("--pipeline", required=True, help="Pipeline family.")
    import_cmd.add_argument("--purpose", required=True, choices=("preview", "final"))
    import_cmd.add_argument("--workflow", required=True, help="Path to the workflow JSON.")

    validate_cmd = commands.add_parser(
        "validate", help="Run the deterministic inspector and record the result."
    )
    validate_cmd.add_argument("--project-dir", required=True)
    validate_cmd.add_argument("--profile-id", required=True)

    test_cmd = commands.add_parser(
        "test-run", help="Record a test-run outcome for the activation gate."
    )
    test_cmd.add_argument("--project-dir", required=True)
    test_cmd.add_argument("--profile-id", required=True)
    test_cmd.add_argument("--success", action="store_true", help="Record a successful test-run.")
    test_cmd.add_argument("--fail", action="store_true", help="Record a failed test-run.")
    test_cmd.add_argument("--detail", default="", help="Free-form test-run detail.")

    activate_cmd = commands.add_parser(
        "activate", help="Activate (gated by validation AND successful test-run)."
    )
    activate_cmd.add_argument("--project-dir", required=True)
    activate_cmd.add_argument("--profile-id", required=True)

    deactivate_cmd = commands.add_parser(
        "deactivate", help="Deactivate an active profile."
    )
    deactivate_cmd.add_argument("--project-dir", required=True)
    deactivate_cmd.add_argument("--profile-id", required=True)

    list_cmd = commands.add_parser(
        "list", help="List imported workflow profiles and their state."
    )
    list_cmd.add_argument("--project-dir", required=True)


def _store_from_args(args: argparse.Namespace) -> tuple[WorkflowImportStore, str]:
    project_dir = Path(args.project_dir)
    if not project_dir.is_dir():
        raise ValueError(f"project directory not found: {project_dir}")
    project_id = project_dir.resolve().name
    return WorkflowImportStore(projects_root=project_dir.parent), project_id


def run_workflow_import_command(args: argparse.Namespace, *, console: Console | None = None) -> int:
    """Dispatch a workflow-import subcommand without contacting ComfyUI."""
    output = ReporterConsole(console or Console())
    try:
        store, project_id = _store_from_args(args)
        command = args.workflow_import_command
        if command == "import":
            return _import(store, project_id, args, output)
        if command == "validate":
            return _validate(store, project_id, args, output)
        if command == "test-run":
            return _test_run(store, project_id, args, output)
        if command == "activate":
            return _activate(store, project_id, args, output)
        if command == "deactivate":
            return _deactivate(store, project_id, args, output)
        if command == "list":
            return _list(store, project_id, output)
        raise ValueError(f"unknown workflow-import command: {command}")
    except (OSError, TypeError, ValueError, WorkflowImportError) as exc:
        output.print(f"[red]Workflow import failed:[/red] {exc}")
        return 1


def _import(store: WorkflowImportStore, project_id: str, args: argparse.Namespace, output: ReporterConsole) -> int:
    graph = Path(args.workflow).read_text(encoding="utf-8")
    profile = store.import_workflow(
        project_id=project_id,
        profile_id=args.profile_id,
        pipeline=args.pipeline,
        purpose=args.purpose,
        graph=graph,
    )
    output.print(f"Imported {profile.profile_id} ({profile.workflow_sha256[:16]}) as draft")
    return 0


def _validate(store: WorkflowImportStore, project_id: str, args: argparse.Namespace, output: ReporterConsole) -> int:
    profile = store.get_profile(project_id, args.profile_id)
    spec = spec_for(profile.pipeline)
    if spec is None:
        raise WorkflowImportError(f"no boundary spec for pipeline {profile.pipeline!r}")
    payload = store.snapshot_bytes(project_id, args.profile_id)
    report = validate_workflow(payload, spec=spec)
    analysis = inspect_workflow(payload, spec=spec)
    store.record_validation(project_id, args.profile_id, valid=report.valid, analysis=analysis)
    state = store.get_profile(project_id, args.profile_id).status
    output.print(f"Validation {'PASS' if report.valid else 'FAIL'}; state now {state}")
    return 0 if report.valid else 2


def _test_run(store: WorkflowImportStore, project_id: str, args: argparse.Namespace, output: ReporterConsole) -> int:
    if args.success == args.fail:
        raise ValueError("exactly one of --success / --fail is required")
    result = TestRunResult(success=args.success, detail=args.detail)
    store.record_test_run(project_id, args.profile_id, result)
    state = store.get_profile(project_id, args.profile_id).status
    output.print(f"Test-run recorded ({'pass' if result.success else 'fail'}); state now {state}")
    return 0 if result.success else 2


def _activate(store: WorkflowImportStore, project_id: str, args: argparse.Namespace, output: ReporterConsole) -> int:
    store.activate(project_id, args.profile_id)
    output.print(f"{args.profile_id} is now active")
    return 0


def _deactivate(store: WorkflowImportStore, project_id: str, args: argparse.Namespace, output: ReporterConsole) -> int:
    store.deactivate(project_id, args.profile_id)
    output.print(f"{args.profile_id} deactivated")
    return 0


def _list(store: WorkflowImportStore, project_id: str, output: ReporterConsole) -> int:
    profiles = store.list_profiles(project_id)
    if not profiles:
        output.print("No imported workflow profiles.")
        return 0
    table = Table("Profile", "Pipeline", "Purpose", "State", "Snapshot")
    for profile in profiles:
        table.add_row(
            profile.profile_id,
            profile.pipeline,
            profile.purpose,
            profile.status,
            profile.workflow_sha256[:16],
        )
    output.print(table)
    return 0
