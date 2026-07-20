from __future__ import annotations

import argparse

from rich.console import Console

from feverslop.application.generate_render_plan import GenerateRenderPlanRequest
from feverslop.composition.generate_render_plan import execute_generate_render_plan
from feverslop.domain.ltx_rendering import ROLLING_FRAME_PROFILES
from feverslop.path_utils import coerce_local_path


console = Console()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        required=True,
        help="Path to the project config.json.",
    )
    parser.add_argument(
        "--app-config",
        default="app_config.json",
        help="Path to global app_config.json. If missing, defaults are used.",
    )
    parser.add_argument(
        "--render-storyboard",
        action="store_true",
        help="Render Z-Image startframes for all scenes.",
    )
    parser.add_argument(
        "--zimage-workflow",
        default=None,
        help="Path to ComfyUI Z-Image workflow API JSON.",
    )
    parser.add_argument(
        "--concept-batch-size",
        type=int,
        default=0,
        help="Generate concept prompts in batches of N segments. 0 disables batching.",
    )
    parser.add_argument(
        "--video-workflow",
        action="append",
        default=[],
        help="Video workflow path used to clamp generated scene durations. Repeat as needed.",
    )
    parser.add_argument(
        "--rolling-frame-profile",
        choices=sorted(ROLLING_FRAME_PROFILES),
        default="original",
        help="Rolling-frame overhead profile used to clamp generated scene durations.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
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


if __name__ == "__main__":
    main()
