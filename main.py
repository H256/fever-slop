from __future__ import annotations

from pathlib import Path
import argparse

from rich.console import Console

from autoprompter.application.generate_render_plan import GenerateRenderPlanRequest
from autoprompter.composition.generate_render_plan import build_generate_render_plan_use_case


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
    return parser


def main():
    args = build_arg_parser().parse_args()
    build_generate_render_plan_use_case(console=console).execute(
        GenerateRenderPlanRequest(
            project_config_path=Path(args.project),
            app_config_path=Path(args.app_config),
            concept_batch_size=int(args.concept_batch_size or 0),
            render_storyboard=bool(args.render_storyboard),
            zimage_workflow_path=Path(args.zimage_workflow) if args.zimage_workflow else None,
        )
    )


if __name__ == "__main__":
    main()
