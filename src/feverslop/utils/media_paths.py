"""Shared filesystem conventions for generated media artifacts."""

from __future__ import annotations

import re
from pathlib import Path


def safe_file_stem(value: str | None, fallback: str) -> str:
    """Return an ASCII filename stem without path or shell-sensitive characters."""
    raw = str(value or "").strip() or str(fallback or "").strip() or "fallback"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    return safe or "fallback"


def write_concat_list(video_files: list[Path], output_file: str | Path) -> Path:
    """Write an ffmpeg concat manifest and create its parent directory."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for path in video_files:
        absolute = Path(path).resolve()
        escaped = absolute.as_posix().replace("'", r"'\''")
        lines.append(f"file '{escaped}'")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
