from __future__ import annotations

from dataclasses import replace

from feverslop.domain.movie import CinematicShot, MovieBible, MovieScreenplayArtifact, StoryArch
from feverslop.adapters.movie_planning_helpers import (
    _dialogue_actor_ids,
    _ensure_minimum_actors,
    _normalize_movie_shots,
    _safe_id,
)
from feverslop.adapters.movie_planning_bible import _movie_bible_from_data, _parse_screenplay_beat


class DeterministicMoviePlanner:
    def generate_story_arch(self, *, title: str, source_type: str, story_text: str, desired_length: float) -> StoryArch:
        text = " ".join(story_text.strip().split())
        from feverslop.adapters.movie_planning_bible import _split_beats, _split_screenplay_beats
        beats = _split_screenplay_beats(story_text) if source_type == "screenplay" else _split_beats(text)
        return StoryArch(title=title, premise=text, beats=tuple(beats))

    def generate_movie_bible(self, *, title: str, source_type: str, story_text: str, desired_length: float, story_arch: StoryArch, config: dict) -> MovieBible:
        return _movie_bible_from_data(
            {},
            title=title,
            source_type=source_type,
            source_text=story_text,
            story_arch=story_arch,
            config=config,
            desired_length=desired_length,
        )

    def generate_movie_continuity_plan(self, **_kwargs) -> dict:
        return {}

    def generate_movie_story_design(self, **_kwargs) -> dict:
        return {}

    def generate_movie_screenplay(self, **_kwargs) -> dict:
        return {}

    def generate_movie_narrative_plan(self, **_kwargs) -> dict:
        return {}

    def plan_shots_from_bible(
        self,
        *,
        bible: MovieBible,
        screenplay: MovieScreenplayArtifact | None = None,
        desired_length: float,
        width: int,
        height: int,
        min_duration: float = 4.0,
        max_duration: float = 20.0,
    ) -> tuple[CinematicShot, ...]:
        shots = self.plan_shots(
            story_arch=bible.story_arch,
            desired_length=desired_length,
            width=width,
            height=height,
            min_duration=min_duration,
            max_duration=max_duration,
        )
        default_actor = bible.actors[0].id if bible.actors else "main_character"
        default_location = bible.locations[0].id if bible.locations else "primary_location"
        default_location_name = bible.locations[0].name if bible.locations else "Primary Location"
        return tuple(
            replace(
                shot,
                actor_ids=shot.actor_ids or (default_actor,),
                location_id=shot.location_id or default_location,
                location=default_location_name if not shot.location or shot.location == "story-consistent cinematic location" else shot.location,
            )
            for shot in shots
        )

    def plan_shots(
        self,
        *,
        story_arch: StoryArch,
        desired_length: float,
        width: int,
        height: int,
        min_duration: float = 4.0,
        max_duration: float = 20.0,
    ) -> tuple[CinematicShot, ...]:
        beats = story_arch.beats or (story_arch.premise,)
        duration = max(1.0, float(desired_length) / len(beats))
        shots = []
        for index, beat in enumerate(beats, start=1):
            screenplay = _parse_screenplay_beat(beat)
            if screenplay is not None:
                description = screenplay["action"] or screenplay["dialogue"] or screenplay["location"]
                shots.append(
                    CinematicShot(
                        shot_id=f"shot_{index:04}",
                        description=description,
                        duration_seconds=duration,
                        camera=screenplay["camera"],
                        action=screenplay["action"],
                        expression=screenplay["expression"],
                        location=screenplay["location"],
                        dialogue=screenplay["dialogue"],
                        actor_ids=tuple(_dialogue_actor_ids(screenplay["dialogue"])),
                        location_id=_safe_id(screenplay["location"]),
                    )
                )
                continue
            shots.append(
                CinematicShot(
                    shot_id=f"shot_{index:04}",
                    description=beat,
                    duration_seconds=duration,
                    camera="slow dolly with motivated cinematic framing",
                    action=beat,
                    expression="subtle emotionally grounded acting",
                    location="story-consistent cinematic location",
                    actor_ids=(),
                    location_id="",
                )
            )
        shots = _ensure_minimum_actors(shots, story_arch)
        return _normalize_movie_shots(
            shots,
            desired_length=float(desired_length),
            min_duration=float(min_duration),
            max_duration=float(max_duration),
        )
