from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class DurationCapability:
    fps: int
    min_seconds: float
    max_seconds: float
    preferred_seconds: float
    frame_alignment: int = 1
    frame_offset: int = 0

    @classmethod
    def create(
        cls, *, fps: int, min_seconds: float, max_seconds: float,
        preferred_seconds: float, frame_alignment: int = 1, frame_offset: int = 0,
    ) -> "DurationCapability":
        if type(fps) is not int or fps <= 0 or type(frame_alignment) is not int or frame_alignment <= 0:
            raise ValueError("fps and frame_alignment must be positive integers")
        values = (float(min_seconds), float(max_seconds), float(preferred_seconds))
        if not all(isfinite(value) for value in values) or not (0 < values[0] <= values[2] <= values[1]):
            raise ValueError("duration limits must satisfy 0 < min <= preferred <= max")
        if type(frame_offset) is not int or frame_offset < 0:
            raise ValueError("frame_offset must be a non-negative integer")
        return cls(fps, *values, frame_alignment, frame_offset)

    def frames_for(self, seconds: float) -> int:
        value = float(seconds)
        if not isfinite(value) or value < self.min_seconds or value > self.max_seconds:
            raise ValueError(
                f"duration {seconds!r} outside profile limits "
                f"[{self.min_seconds}, {self.max_seconds}] seconds",
            )
        raw = max(1, round(value * self.fps))
        raw = self.frame_offset + round((raw - self.frame_offset) / self.frame_alignment) * self.frame_alignment
        if raw > round(self.max_seconds * self.fps):
            raw -= self.frame_alignment
        if raw <= 0 or raw < round(self.min_seconds * self.fps):
            raise ValueError("duration cannot be represented with the profile frame alignment")
        return raw

    def validate(self, seconds: float) -> bool:
        self.frames_for(seconds)
        return True

    def to_dict(self) -> dict[str, float | int]:
        return {
            "fps": self.fps,
            "min_seconds": self.min_seconds,
            "max_seconds": self.max_seconds,
            "preferred_seconds": self.preferred_seconds,
            "frame_alignment": self.frame_alignment,
            "frame_offset": self.frame_offset,
        }
