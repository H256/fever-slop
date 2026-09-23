from __future__ import annotations

import argparse


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
