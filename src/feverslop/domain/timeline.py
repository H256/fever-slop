from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TimelineSegment:
    """Represents a segment on a vocal timeline."""

    start: float
    end: float
    kind: str  # "vocals" or "instrumental"
    text: str = ""
    word_timestamps: tuple[dict[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError("Timeline segment bounds must be finite")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Timeline segment must have 0 <= start < end")
