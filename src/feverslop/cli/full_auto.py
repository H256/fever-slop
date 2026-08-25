"""Canonical Full-Auto command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from feverslop.adapters.pipeline_runner_options import add_runner_options, runner_options_from_args
from feverslop.application.full_auto import FullAutoRequest
from feverslop.composition.full_auto import build_full_auto_use_case
from feverslop.path_utils import coerce_local_path

console = Console()


def parse_optional_bool(value: str | None) -> bool:
    if value is None:
        return True
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a full FeverSlop project from an idea and style.")
    add_full_auto_args(parser)
    return parser


def add_full_auto_args(parser: argparse.ArgumentParser) -> None:
    """Add Full-Auto options to a standalone or unified CLI parser."""
    parser.add_argument("--idea", required=True)
    parser.add_argument("--style", required=True)
    parser.add_argument("--music-style", default=None)
    parser.add_argument("--project-name", default=None)
    parser.add_argument("--projects-dir", default="projects")
    parser.add_argument("--workflow", default=str(Path("workflows") / "audio_song_v2.json"))
    parser.add_argument("--duration-seconds", type=float, default=120.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--fps", type=int, choices=[16, 24, 50], default=24)
    parser.add_argument("--language", default="en")
    parser.add_argument("--bpm", type=int, default=None)
    parser.add_argument("--keyscale", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-video-pipeline", action="store_true")
    parser.add_argument(
        "--silent-mode",
        nargs="?",
        const=True,
        default=False,
        type=parse_optional_bool,
        help="Disable singing, lip-sync, and vocal performance prompts while preserving emotional acting.",
    )
    add_runner_options(parser)


def request_from_args(args: argparse.Namespace) -> FullAutoRequest:
    return FullAutoRequest(
        idea=args.idea,
        style=args.style,
        music_style=args.music_style,
        project_name=args.project_name,
        projects_dir=Path(args.projects_dir),
        duration_seconds=float(args.duration_seconds),
        width=int(args.width),
        height=int(args.height),
        fps=int(args.fps),
        language=args.language,
        bpm=args.bpm,
        keyscale=args.keyscale,
        seed=int(args.seed),
        silent_mode=bool(args.silent_mode),
        run_video_pipeline=bool(args.run_video_pipeline),
        runner_options=runner_options_from_args(args),
    )


def run_full_auto_command(args: argparse.Namespace, *, output: Console = console) -> None:
    build_full_auto_use_case(
        app_config_path=coerce_local_path(args.app_config),
        workflow_path=coerce_local_path(args.workflow),
        console=output,
    ).execute(request_from_args(args))


def main() -> None:
    run_full_auto_command(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
