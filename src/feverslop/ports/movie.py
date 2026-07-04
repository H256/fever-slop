from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol

from feverslop.domain.movie import CinematicShot, MovieBible, MovieContinuityPlan, StoryArch


class StoryGenerationPort(Protocol):
    def generate_story_arch(self, *, title: str, source_type: str, story_text: str, desired_length: float) -> StoryArch:
        """Create a structured movie story arch from prose or screenplay text."""


class ScenePlanningPort(Protocol):
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
        """Create a continuous cinematic shot plan."""

    def generate_movie_bible(self, *, title: str, source_type: str, story_text: str, desired_length: float, story_arch: StoryArch, config: dict) -> MovieBible:
        """Create the movie bible that constrains actors, locations, continuity, and style."""

    def generate_movie_continuity_plan(
        self,
        *,
        title: str,
        source_type: str,
        story_text: str,
        desired_length: float,
        bible: MovieBible,
        shots: tuple[CinematicShot, ...],
        config: dict,
    ) -> MovieContinuityPlan | dict:
        """Create a causal continuity ledger for movie shots."""

    def plan_shots_from_bible(
        self,
        *,
        bible: MovieBible,
        desired_length: float,
        width: int,
        height: int,
        min_duration: float = 4.0,
        max_duration: float = 20.0,
    ) -> tuple[CinematicShot, ...]:
        """Create a shot plan constrained to movie bible actor and location ids."""


class VisualGenerationPort(Protocol):
    def render_movie(
        self,
        *,
        project_dir: Path,
        render_plan_path: Path,
        selected_scenes: list[int] | None = None,
        concat_only: bool = False,
        on_clip_rendered: Callable[[int, int, int], None] | None = None,
    ) -> Path:
        """Render a final movie file from a movie render plan."""


class ReferenceGenerationPort(Protocol):
    def generate(self, *, project_dir: Path) -> Path:
        """Render/fill movie actor and location reference sheets."""
