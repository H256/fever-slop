"""Explicit argument-definition helpers shared by legacy CLI entry points."""

from __future__ import annotations

import argparse

from feverslop.domain.ltx_rendering import ROLLING_FRAME_PROFILES


def add_render_args(parser: argparse.ArgumentParser, *, project_required: bool) -> None:
    """Add the render-plan options used by subcommand and legacy forms."""
    parser.add_argument(
        "--project",
        required=project_required,
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
        default=False,
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
