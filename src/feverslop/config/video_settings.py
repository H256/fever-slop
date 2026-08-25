from dataclasses import dataclass


@dataclass(frozen=True)
class VideoSettings:
    fps: int = 24
    width: int = 1280
    height: int = 704
    megapixels: float | None = None

    def seconds_to_frame(self, seconds: float) -> int:
        return round(seconds * self.fps)

    def scene_frame_count(self, duration_seconds: float) -> int:
        return max(1, self.seconds_to_frame(duration_seconds))

    def scene_frame_count_between(self, start_seconds: float, end_seconds: float) -> int:
        return max(1, self.seconds_to_frame(end_seconds) - self.seconds_to_frame(start_seconds))
