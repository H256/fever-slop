from __future__ import annotations

from pathlib import Path
from typing import Protocol

from feverslop.domain.movie import CinematicShot, StoryArch


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


class VisualGenerationPort(Protocol):
    def render_movie(self, *, project_dir: Path, render_plan_path: Path) -> Path:
        """Render a final movie file from a movie render plan."""


class ReferenceGenerationPort(Protocol):
    def generate(self, *, project_dir: Path) -> Path:
        """Render/fill movie actor and location reference sheets."""
