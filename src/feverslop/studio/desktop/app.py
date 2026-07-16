from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FeverSlop native Studio")
    parser.add_argument("--projects-root", type=Path, default=Path("projects"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    from feverslop.studio.desktop.runtime import run_studio

    args = parse_args(argv)
    return run_studio(args.projects_root)

