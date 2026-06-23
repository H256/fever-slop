from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from feverslop.adapters.pipeline_runner_options import add_runner_options, runner_options_from_args
from feverslop.application.full_auto import FullAutoRequest
from feverslop.composition.full_auto import build_full_auto_use_case
from feverslop.path_utils import coerce_local_path


console = Console()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a full FeverSlop project from an idea and style.")
    parser.add_argument("--idea", required=True)
    parser.add_argument("--style", required=True)
    parser.add_argument("--project-name", default=None)
    parser.add_argument("--projects-dir", default="projects")
    parser.add_argument("--workflow", default=str(Path("workflows") / "audio_song.json"))
    parser.add_argument("--duration-seconds", type=float, default=120.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--language", default="en")
    parser.add_argument("--bpm", type=int, default=None)
    parser.add_argument("--keyscale", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-video-pipeline", action="store_true")
    add_runner_options(parser)
    return parser


def request_from_args(args: argparse.Namespace) -> FullAutoRequest:
    return FullAutoRequest(
        idea=args.idea,
        style=args.style,
        project_name=args.project_name,
        projects_dir=Path(args.projects_dir),
        duration_seconds=float(args.duration_seconds),
        width=int(args.width),
        height=int(args.height),
        language=args.language,
        bpm=args.bpm,
        keyscale=args.keyscale,
        seed=int(args.seed),
        run_video_pipeline=bool(args.run_video_pipeline),
        runner_options=runner_options_from_args(args),
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    build_full_auto_use_case(
        app_config_path=coerce_local_path(args.app_config),
        workflow_path=coerce_local_path(args.workflow),
        console=console,
    ).execute(request_from_args(args))


if __name__ == "__main__":
    main()
