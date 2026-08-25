"""Legacy facade for the packaged scene-SRT repair CLI."""

import os

from feverslop.tools import repair_scene_srt as _cli
from rich.console import Console

__all__ = ["ensure_output_writable", "main"]
console = Console()


def ensure_output_writable(output_srt):
    _cli.os = os
    return _cli.ensure_output_writable(output_srt)


def main():
    _cli.os = os
    _cli.console = console
    return _cli.main()

if __name__ == "__main__":
    main()
