from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

from feverslop.domain.prompt_revisions import PromptField


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def run_revisions(args: argparse.Namespace) -> None:
    """Handle the revisions subcommand."""
    from rich.console import Console

    from feverslop.application.prompt_revisions import (
        LoadPromptHistoryUseCase,
        PatchPromptError,
        RestoreRevisionUseCase,
    )
    from feverslop.infra.sqlite_adapter import SqliteRevisionStore

    console = Console()
    project_dir = Path(args.project_dir)

    # Resolve revision store from project dir
    db_path = project_dir / "output" / "revision_store.db"
    if not db_path.exists():
        # Create parent dir if needed
        db_path.parent.mkdir(parents=True, exist_ok=True)

    store = SqliteRevisionStore(str(db_path))

    if args.restore:
        # Restore mode
        restore = RestoreRevisionUseCase(store=store, clock=_utc_now)
        try:
            restored = restore.execute(
                project_id=str(project_dir.name),
                scene_number=args.scene,
                field=PromptField(args.field),
                revision_id=args.restore,
            )
            console.print(f"[green]Restored revision {restored.id}[/]\n")
        except (ValueError, PatchPromptError) as exc:
            console.print(f"[red]Error: {exc}[/]")
            sys.exit(1)

    # Load and display history
    loader = LoadPromptHistoryUseCase(store=store)
    result = loader.execute(
        project_id=str(project_dir.name),
        scene_number=args.scene,
        field=PromptField(args.field),
    )

    if not result.history.revisions:
        console.print(f"No revisions for scene {args.scene} ({args.field})")
        return

    console.print(f"\n[bold]Scene {args.scene} - {args.field}[/]\n")
    console.print(f"Available fields: {', '.join(f.value for f in result.available_fields)}\n")

    for i, rev in enumerate(reversed(result.history.revisions[-args.limit:])):
        marker = "(restored)" if rev.restored_from else ""
        console.print(
            f"{rev.id[:8]}  {rev.created_at.strftime('%Y-%m-%d %H:%M')}  {marker}",
        )
        console.print(f"  {rev.value}")
        if rev.parent_id:
            console.print(f"  parent: {rev.parent_id[:8]}")
        if rev.restored_from:
            console.print(f"  restored from: {rev.restored_from[:8]}")
        console.print("")


def run_rebuild_preview(args: argparse.Namespace) -> None:
    """Handle the rebuild-preview subcommand."""
    from rich.console import Console

    from feverslop.application.rebuild_preview import PreviewRebuildUseCase
    from feverslop.domain.rebuild_policy import (
        ChangeSet,
    )

    console = Console()
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
