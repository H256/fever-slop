from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class AudioTimingWindow:
    """Absolute audio interval carried into one generated continuation segment."""

    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if not isfinite(self.start_seconds) or not isfinite(self.end_seconds):
            raise ValueError("audio timing must be finite")
        if self.start_seconds < 0 or self.end_seconds <= self.start_seconds:
            raise ValueError("audio timing window is invalid")

    @property
    def duration_seconds(self) -> float:
        return round(self.end_seconds - self.start_seconds, 6)
