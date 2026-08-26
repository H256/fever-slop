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


def validate_audio_timing_windows(windows: list[AudioTimingWindow], *, song_duration: float) -> None:
    if not isfinite(song_duration) or song_duration <= 0:
        raise ValueError("song_duration must be positive")
    previous_end = 0.0
    for window in windows:
        if window.start_seconds < previous_end:
            raise ValueError("audio timing windows overlap or are out of order")
        if window.end_seconds > song_duration + 1e-6:
            raise ValueError("audio timing window exceeds song duration")
        previous_end = window.end_seconds
