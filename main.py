from __future__ import annotations

import argparse
import sys

from rich.console import Console

from feverslop.cli.canonical_plan_migration_cli import (
    build_canonical_plan_migration_parser,
    run_canonical_plan_migration,
)
from feverslop.cli.canonical_plan_cli import (
    build_canonical_plan_parsers,
    run_canonical_plan_command,
)
from feverslop.cli.movie_cli import add_movie_args
from feverslop.cli.run_cli import build_run_parser, run_project_command
from feverslop.cli.revision_commands import run_rebuild_preview, run_revisions
from feverslop.cli.revisions import build_rebuild_preview_parser, build_revisions_parser
from feverslop.cli.shared_args import add_render_args
from feverslop.composition.generate_render_plan import execute_generate_render_plan

console = Console()


def _build_render_parser(subparsers) -> argparse.ArgumentParser:
    """Build argument parser for the render (old render-plan) subcommand."""
    parser = subparsers.add_parser(
        "render",
        help="Run the render-plan pipeline (generate render plan, storyboard, etc.).",
    )
    add_render_args(parser, project_required=True)
    return parser


def _run_render(args: argparse.Namespace) -> None:
    """Execute the old render-plan pipeline."""
    from feverslop.application.generate_render_plan import GenerateRenderPlanRequest
    from feverslop.path_utils import coerce_local_path

    execute_generate_render_plan(
        GenerateRenderPlanRequest(
            project_config_path=coerce_local_path(args.project),
            app_config_path=coerce_local_path(args.app_config),
            concept_batch_size=int(args.concept_batch_size or 0),
            video_workflow_paths=tuple(coerce_local_path(path) for path in args.video_workflow),
            rolling_frame_profile=args.rolling_frame_profile,
            render_storyboard=bool(args.render_storyboard),
            zimage_workflow_path=coerce_local_path(args.zimage_workflow) if args.zimage_workflow else None,
        ),
        console=console,
    )


def _run_movie(args: argparse.Namespace) -> None:
    """Execute the movie pipeline."""
    from feverslop.composition.movie_pipeline import run

    run(args)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the unified argument parser with subcommands.

    Supports ``render`` (old render-plan pipeline) and ``movie`` (movie pipeline).
    When invoked without a subcommand, falls back to the render pipeline for
    backward compatibility.
    """
    parser = argparse.ArgumentParser(
        description="FeverSlop music video pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", description="Sub-command")

    # --- movie subcommand ---
    movie_parser = subparsers.add_parser(
        "movie",
        help="Run movie pipeline stages for an existing FeverSlop movie project.",
    )
    _add_movie_args(movie_parser)

    # --- render subcommand ---
    _build_render_parser(subparsers)

    # --- revisions subcommand ---
    build_revisions_parser(subparsers)

    # --- rebuild-preview subcommand ---
    build_rebuild_preview_parser(subparsers)

    # --- canonical plan migration subcommand ---
    build_canonical_plan_migration_parser(subparsers)
    build_canonical_plan_parsers(subparsers)
    build_run_parser(subparsers)

    # --- backward-compatibility: top-level render-plan args ---
    # When no subcommand is given, these top-level args are parsed and the
    # render pipeline is executed automatically.
    _add_render_args(parser)

    return parser


def _add_movie_args(parser: argparse.ArgumentParser) -> None:
    """Populate a parser with movie-pipeline arguments (thin wrapper around movie_cli)."""
    add_movie_args(parser)


def _add_render_args(parser: argparse.ArgumentParser) -> None:
    """Add render-plan arguments at the top level for backward compatibility."""
    add_render_args(parser, project_required=False)


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.command == "movie":
        _run_movie(args)
    elif args.command == "render":
        _run_render(args)
    elif args.command == "revisions":
        run_revisions(args)
    elif args.command == "rebuild-preview":
        run_rebuild_preview(args)
    elif args.command == "plan-migrate":
        exit_code = run_canonical_plan_migration(args, console=console)
        if exit_code:
            raise SystemExit(exit_code)
    elif args.command in {"plan", "status"}:
        exit_code = run_canonical_plan_command(args, console=console)
        if exit_code:
            raise SystemExit(exit_code)
    elif args.command == "run":
        exit_code = run_project_command(args, console=console)
        if exit_code:
            raise SystemExit(exit_code)
    elif args.project:
        # Backward compatibility: --project at top level => render pipeline.
        _run_render(args)
    else:
        build_arg_parser().print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
