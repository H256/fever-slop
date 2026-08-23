from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from feverslop.domain.movie import (
    CinematicShot,
    MovieBible,
    MovieContinuityPlan,
    MovieNarrativePlan,
    MovieScreenplayArtifact,
    MovieStoryDesign,
    StoryArch,
)


class MovieArtifactWriter(Protocol):
    """Persistence adapter for movie planning artifacts (JSON files, text files, directories)."""

    def write_json(self, path: str | Path, data: Any) -> Path:
        """Serialize *data* as pretty JSON with a trailing newline and write to *path*."""

    def write_text(self, path: str | Path, text: str) -> Path:
        """Write UTF-8 text to *path* (creates parent directories as needed)."""

    def ensure_dir(self, path: str | Path, *, exist_ok: bool = True) -> Path:
        """Ensure a directory exists. Raises if the path already exists as a file."""


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
    ) -> MovieContinuityPlan:
        """Create a causal continuity ledger for movie shots."""

    def generate_movie_screenplay(
        self,
        *,
        title: str,
        source_type: str,
        story_text: str,
        desired_length: float,
        bible: MovieBible,
        story_arch: StoryArch,
        story_design: MovieStoryDesign,
        config: dict,
    ) -> MovieScreenplayArtifact:
        """Create or normalize the canonical persisted movie screenplay."""

    def generate_movie_story_design(
        self,
        *,
        title: str,
        source_type: str,
        story_text: str,
        desired_length: float,
        bible: MovieBible,
        story_arch: StoryArch,
        config: dict,
    ) -> MovieStoryDesign:
        """Create dramaturgical design before canonical screenplay authoring."""

    def generate_movie_narrative_plan(
        self,
        *,
        title: str,
        source_type: str,
        desired_length: float,
        bible: MovieBible,
        screenplay: MovieScreenplayArtifact,
        config: dict,
    ) -> MovieNarrativePlan:
        """Create act/sequence causality and setup/payoff memory from the canonical screenplay."""

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
        """Create a shot plan constrained to movie bible actor and location ids."""


class VisualGenerationPort(Protocol):
    def render_movie(
        self,
        *,
        project_dir: Path,
        render_plan_path: Path,
        selected_scenes: list[int] | None = None,
        concat_only: bool = False,
        continuity_keyframes: str = "none",
        on_clip_rendered: Callable[[int, int, int], None] | None = None,
    ) -> Path:
        """Render a final movie file from a movie render plan."""


class ReferenceGenerationPort(Protocol):
    def generate(self, *, project_dir: Path) -> Path:
        """Render/fill movie actor and location reference sheets."""
