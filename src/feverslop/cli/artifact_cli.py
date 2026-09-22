"""Manual artifact prune CLI (issue #1388, piece 3).

``feverslop artifact prune`` is a manual command that never runs inside the
pipeline. Safe mode (default) scans and prints the machine-readable JSON
report without writing anything to the project tree; ``--apply --archive
PATH`` archives the eligible candidates first, then deletes them and writes
a timestamped machine-readable report under ``output/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from feverslop.application.artifact_prune import PruneReport, apply_prune, scan_project
from feverslop.errors import FeverSlopDataError
from feverslop.ports.reporting import ReporterConsole


def build_artifact_parsers(subparsers) -> None:
    artifact = subparsers.add_parser(
        "artifact",
        help="Manage project artifact lifecycle (manual; never runs in the pipeline).",
    )
    commands = artifact.add_subparsers(dest="artifact_command", required=True)
    prune = commands.add_parser(
        "prune",
        help="Report or delete regenerable cache artifacts (archive-first).",
    )
    prune.add_argument("project")
    mode = prune.add_mutually_exclusive_group()
    mode.add_argument("--safe", action="store_true", help="Scan and report only; never write (default).")
    mode.add_argument("--apply", action="store_true", help="Archive eligible candidates, then delete them.")
    prune.add_argument("--archive", help="Archive path for --apply (required with --apply).")


def run_artifact_prune_command(args: argparse.Namespace, *, console: Console | None = None) -> int:
    output = ReporterConsole(console or Console())
    project = Path(args.project).resolve()
    try:
        if args.apply:
            if not args.archive:
                output.print("[red]--apply requires --archive PATH[/red]")
                return 1
            report = scan_project(project, reporter=output.reporter)
            final = apply_prune(project, args.archive, report, reporter=output.reporter)
            _write_report_file(project, final, output)
            output.print(final.to_json())
            output.print(f"[green]OK[/green] pruned {len(final.deleted)} file(s); archive: {final.archive_path}")
            return 0
        report = scan_project(project, reporter=output.reporter)
        output.print(report.to_json())
        eligible = sum(1 for entry in report.candidates if entry["eligible"])
        if eligible:
            output.print(
                f"[yellow]ACTION REQUIRED[/yellow] {eligible} eligible candidate(s); "
                "re-run with --apply --archive PATH"
            )
            return 2
        output.print("[green]OK[/green] no eligible candidates")
        return 0
    except (FeverSlopDataError, OSError, TypeError, ValueError) as exc:
        output.print(f"[red]Artifact prune failed:[/red] {exc}")
        return 1


def _write_report_file(project: Path, report: PruneReport, output: ReporterConsole) -> None:
    path = project / "output" / f"prune_report_{report.created_at.replace(':', '-')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json(), encoding="utf-8")
    output.reporter.file("prune report", path)
