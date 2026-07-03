from __future__ import annotations

from feverslop.domain.movie import CinematicShot, StoryArch


class DeterministicMoviePlanner:
    def generate_story_arch(self, *, title: str, source_type: str, story_text: str, desired_length: float) -> StoryArch:
        text = " ".join(story_text.strip().split())
        beats = _split_beats(text)
        return StoryArch(title=title, premise=text, beats=tuple(beats))

    def plan_shots(self, *, story_arch: StoryArch, desired_length: float, width: int, height: int) -> tuple[CinematicShot, ...]:
        beats = story_arch.beats or (story_arch.premise,)
        duration = max(1.0, float(desired_length) / len(beats))
        shots = []
        for index, beat in enumerate(beats, start=1):
            shots.append(
                CinematicShot(
                    shot_id=f"shot_{index:04}",
                    description=beat,
                    duration_seconds=duration,
                    camera="slow dolly with motivated cinematic framing",
                    action=beat,
                    expression="subtle emotionally grounded acting",
                    location="story-consistent cinematic location",
                )
            )
        return tuple(shots)


def _split_beats(text: str) -> list[str]:
    parts = [part.strip(" .") for part in text.replace("\n", " ").split(".") if part.strip()]
    return parts[:12] or [text]
