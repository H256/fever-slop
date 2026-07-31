from __future__ import annotations

import argparse

from feverslop.domain.prompt_revisions import PromptField


def build_revisions_parser(subparsers) -> argparse.ArgumentParser:
    """Build argument parser for the revisions subcommand."""
    parser = subparsers.add_parser(
        "revisions",
        help="View and manage prompt revision history.",
    )
    parser.add_argument(
        "--project-dir",
        required=True,
        help="Path to the project directory.",
    )
    parser.add_argument(
        "--scene",
        type=int,
        required=True,
        help="Scene number.",
    )
    parser.add_argument(
        "--field",
        choices=[f.value for f in PromptField],
        default="z_image_prompt",
        help="Prompt field to view (default: z_image_prompt).",
    )
    parser.add_argument(
        "--restore",
        type=str,
        default=None,
        help="Restore revision by ID. Appends a new revision with the restored value.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of revisions to display (default: 50).",
    )
    return parser


def build_rebuild_preview_parser(subparsers) -> argparse.ArgumentParser:
    """Build argument parser for the rebuild-preview subcommand."""
    parser = subparsers.add_parser(
        "rebuild-preview",
        help="Preview which artifacts need rebuild for a given change set.",
    )
    parser.add_argument(
        "--project-dir",
        required=True,
        help="Path to the project directory.",
    )
    return parser
