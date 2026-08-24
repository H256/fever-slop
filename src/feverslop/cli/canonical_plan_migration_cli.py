from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from feverslop.adapters.canonical_plan_store import CanonicalPlanStore
from feverslop.application.canonical_plan_migration import (
    MigrationFinding,
    analyze_canonical_plan_migration,
)
from feverslop.errors import FeverSlopDataError


def build_canonical_plan_migration_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "plan-migrate",
        help="Inspect legacy render-plan edits and migrate proven edits into base.json overrides.",
    )
    parser.add_argument("project", help="Path to the FeverSlop project directory.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Back up analyzed plans and atomically write safe overrides to base.json.",
    )
    return parser


def run_canonical_plan_migration(
    args: argparse.Namespace,
    *,
    console: Console | None = None,
) -> int:
    output = console or Console()
    project = Path(args.project).resolve()
    mode = "Apply" if args.apply else "Dry run"
    output.print(f"[bold]{mode}:[/bold] analyzing canonical and legacy render plans in {project}")
    try:
        store = CanonicalPlanStore(project)
        report = analyze_canonical_plan_migration(store.load())
    except (FeverSlopDataError, OSError, ValueError) as exc:
        output.print(f"[red]Cannot analyze render plans:[/red] {exc}")
        return 1

    output.print(
        f"Found {len(report.importable)} importable, "
        f"{len(report.unresolved)} unresolved, and {len(report.no_op)} already applied value(s).",
    )
    for finding in report.findings:
        _print_finding(output, finding)

    if report.unresolved:
        output.print("[red]Blocked:[/red] resolve every unresolved finding before applying.")
        return 2
    if not args.apply:
        output.print("Dry run complete; no files were written. Re-run with --apply to migrate.")
        return 0

    try:
        result = store.apply(report)
    except (FeverSlopDataError, OSError, ValueError) as exc:
        output.print(f"[red]Migration failed:[/red] {exc}")
        return 1
    if not result.applied:
        output.print("No changes needed; base.json already contains every proven override.")
        return 0
    output.print(f"[green]Imported {result.imported_count} override(s) into base.json.[/green]")
    output.print(f"Backup: {result.backup_dir}")
    return 0


def _print_finding(console: Console, finding: MigrationFinding) -> None:
    identity = finding.scene_id or finding.segment_id or (
        f"scene {finding.scene_number}" if finding.scene_number is not None else "artifact"
    )
    details = [finding.kind.upper(), finding.source_path, identity]
    if finding.role:
        details.append(finding.role)
    if finding.field_path:
        details.append(finding.field_path)
    details.append(finding.reason)
    console.print("  " + " | ".join(details))
