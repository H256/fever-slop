from __future__ import annotations

import argparse

from feverslop.ports.reporting import ReporterConsole
from rich.console import Console


def run_rebuild_preview(args: argparse.Namespace) -> None:
    """Handle the rebuild-preview subcommand."""
    from feverslop.application.rebuild_preview import PreviewRebuildUseCase
    from feverslop.domain.rebuild_policy import (
        ChangeSet,
    )

    console = ReporterConsole(Console())
    console.print("[bold]Rebuild Preview[/]\n")

    # Placeholder: load scene documents and compare prompt hashes against provenance
    # For now, demonstrate with an empty change set
    use_case = PreviewRebuildUseCase()
    change = ChangeSet.empty()
    result = use_case.execute(change=change)

    if not result.stages:
        console.print("No rebuild needed - all artifacts current.")
        return

    for stage in result.stages:
        console.print(f"[bold]{stage.value}[/]")

    console.print("\nReusable artifacts:")
    for artifact_state in result.reusable_artifacts:
        console.print(f"  {artifact_state.kind.value} - {artifact_state.state.value}")

    console.print("\nStale artifacts:")
    for artifact_state in result.stale_artifacts:
        console.print(f"  {artifact_state.kind.value} - {artifact_state.state.value}")
