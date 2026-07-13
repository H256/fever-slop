from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimelineSegment:
    """Represents a segment on a vocal timeline."""
    start: float
    end: float
    kind: str  # "vocals" or "instrumental"
    text: str = ""
