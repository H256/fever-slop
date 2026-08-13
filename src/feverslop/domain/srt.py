from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class SrtBlock:
    """Parsed SRT subtitle block."""
    index: int
    start: float
    end: float
    text: str = ""


@dataclass(frozen=True)
class SrtScene:
    """Represents a scene from an SRT with scene number."""
    scene: int
    start: float
    end: float
    text: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_srt_timestamp(value: str) -> float:
    """Parse SRT timestamp 'HH:MM:SS,mmm' to seconds.

    Raises:
        ValueError: If timestamp format is invalid.
    """
    value = value.strip()
    hours, minutes, rest = value.split(":")
    seconds, millis = rest.split(",")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis) / 1000.0
    )


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp 'HH:MM:SS,mmm'."""
    seconds = max(0.0, float(seconds))
    millis_total = round(seconds * 1000)
    millis = millis_total % 1000
    total_seconds = millis_total // 1000
    sec = total_seconds % 60
    total_minutes = total_seconds // 60
    minute = total_minutes % 60
    hour = total_minutes // 60
    return f"{hour:02}:{minute:02}:{sec:02},{millis:03}"


def parse_srt_text(text: str) -> list[SrtBlock]:
    """Parse SRT text into blocks without filesystem access.

    Handles empty files, malformed index lines, and missing timestamps.
    Returns empty list for empty/invalid files.
    """
    text = str(text).strip()
    if not text:
        return []

    blocks = re.split(r"\n\s*\n", text)
    result: list[SrtBlock] = []

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        try:
            index = int(lines[0])
        except ValueError:
            continue

        match = re.match(r"(.+?)\s*-->\s*(.+)", lines[1])
        if not match:
            continue

        start = parse_srt_timestamp(match.group(1))
        end = parse_srt_timestamp(match.group(2))
        body = "\n".join(lines[2:]) if len(lines) > 2 else ""

        result.append(SrtBlock(index=index, start=start, end=end, text=body))

    return result


def parse_srt_blocks(path: str | Path) -> list[SrtBlock]:
    """Compatibility file reader; callers at the adapter boundary should use ``parse_srt_text``."""
    return parse_srt_text(Path(path).read_text(encoding="utf-8"))
