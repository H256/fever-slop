"""CLI commands for inspecting and validating video workflow profiles."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from feverslop.config.app_config import AppConfig


def build_profiles_parser(subparsers) -> None:
    """Register the read-only video workflow profile commands."""
    profiles = subparsers.add_parser(
        "profiles",
        help="List and preflight configured video workflow profiles.",
    )
    commands = profiles.add_subparsers(dest="profile_command", required=True)

    list_parser = commands.add_parser("list", help="List profiles grouped by pipeline and purpose.")
    list_parser.add_argument(
        "--app-config",
        default="app_config.json",
        help="Path to global app_config.json.",
    )

    preflight_parser = commands.add_parser(
        "preflight",
        help="Resolve and validate a requested pipeline/purpose before rendering.",
    )
    preflight_parser.add_argument(
        "--app-config",
        default="app_config.json",
        help="Path to global app_config.json.",
    )
    preflight_parser.add_argument("--pipeline", required=True, help="Video pipeline family.")
    preflight_parser.add_argument("--purpose", required=True, choices=("preview", "final"))
    preflight_parser.add_argument(
        "--profile",
        default=None,
        help="Named profile to validate; omit to resolve the configured default.",
    )


def run_profiles_command(args: argparse.Namespace, *, console: Console | None = None) -> int:
    """Run a profile command without contacting ComfyUI."""
    output = console or Console()
    try:
        config = AppConfig.load(Path(args.app_config))
        if args.profile_command == "list":
            return _list_profiles(config, output)
        return _preflight_profile(config, args, output)
    except (OSError, TypeError, ValueError) as exc:
        output.print(f"[red]Profile preflight failed:[/red] {exc}")
        return 1


def _list_profiles(config: AppConfig, output: Console) -> int:
    table = Table("Pipeline", "Purpose", "Name", "Default")
    for profile in sorted(
        config.video_workflow_profiles,
        key=lambda item: (item.pipeline, item.purpose, item.name),
    ):
        default = config.resolve_video_workflow_profile(
            pipeline=profile.pipeline,
            purpose=profile.purpose,
        )
        table.add_row(
            profile.pipeline,
            profile.purpose,
            profile.name,
            "DEFAULT" if default is profile else "",
        )
    output.print(table)
    return 0


def _preflight_profile(config: AppConfig, args: argparse.Namespace, output: Console) -> int:
    requested = args.profile or "<default>"
    output.print(f"Requested profile: {requested}")
    output.print(f"Pipeline/purpose: {args.pipeline}/{args.purpose}")
    profile = config.resolve_video_workflow_profile(
        pipeline=args.pipeline,
        purpose=args.purpose,
        name=args.profile,
    )
    if profile is None:
        raise ValueError(
            f"No video workflow profile is configured for {args.pipeline}/{args.purpose}"
        )
    output.print(f"Resolved profile: {profile.name}")
    output.print(f"Workflow: {profile.workflow_path}")
    output.print(f"Stages: {profile.stages}; output scale: {profile.output_scale:g}")
    return 0
