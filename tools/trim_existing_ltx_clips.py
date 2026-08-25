"""Legacy facade for the packaged LTX clip-trimming CLI."""

import shutil

from feverslop.adapters.video_postprocessor import VideoPostProcessor
from feverslop.tools import trim_existing_ltx_clips as _cli
from feverslop.utils.rich_progress import build_progress
from rich.console import Console

__all__ = ["_resolve_ffmpeg_path", "main"]
console = Console()


def _sync_legacy_seams():
    _cli.shutil = shutil
    _cli.VideoPostProcessor = VideoPostProcessor
    _cli.build_progress = build_progress
    _cli.console = console


def _resolve_ffmpeg_path(value: str) -> str:
    _sync_legacy_seams()
    return _cli._resolve_ffmpeg_path(value)


def main():
    _sync_legacy_seams()
    return _cli.main()

if __name__ == "__main__":
    main()
