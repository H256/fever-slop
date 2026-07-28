from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from feverslop.domain.screenplay import looks_like_screenplay


class MovieInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    source_type: str
    story_text: str
    desired_length: float
    width: int = 1280
    height: int = 704
    mode: str = "scaffold"
    min_scene_duration: float = 4.0
    max_scene_duration: float = 20.0
    config: dict = Field(default_factory=dict)

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, v: str) -> str:
        if v not in {"short_story", "screenplay"}:
            raise ValueError("source_type must be short_story or screenplay")
        return v

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in {"scaffold", "full_auto"}:
            raise ValueError("movie mode must be scaffold or full_auto")
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Movie project name is required")
        return v

    @field_validator("story_text")
    @classmethod
    def validate_story_text(cls, v: str) -> str:
        if len(v.strip()) < 20:
            raise ValueError("Movie story input is too short")
        return v

    @field_validator("desired_length")
    @classmethod
    def validate_desired_length(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("desired_length must be positive")
        return v

    @field_validator("width", "height")
    @classmethod
    def validate_resolution(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("resolution width and height must be positive")
        return v

    @model_validator(mode="after")
    def validate_screenplay(self) -> MovieInput:
        if self.source_type == "screenplay" and not looks_like_screenplay(self.story_text):
            raise ValueError("screenplay input must contain scene headings such as INT. or EXT.")
        return self


@dataclass(frozen=True)
class MovieScaffoldResult:
    project_slug: str
    project_dir: Path
    bible_path: Path
    story_arch_path: Path
    render_plan_path: Path
    reference_manifest_path: Path
    story_design_path: Path | None = None
    screenplay_path: Path | None = None
    narrative_plan_path: Path | None = None
    scene_cards_path: Path | None = None
    shot_cards_path: Path | None = None


@dataclass(frozen=True)
class MovieProductionResult(MovieScaffoldResult):
    final_video_path: Path | None = None


def _planner_source_text(request: MovieInput, config: dict) -> str:
    parts = [request.story_text]
    for label, value in [
        ("story_idea", config.get("story_idea")),
        ("style", config.get("style")),
        ("subject", config.get("subject")),
        ("steering", config.get("steering")),
        ("prompt_guidance", config.get("prompt_guidance")),
        ("dialogue_language", config.get("dialogue_language")),
        ("actors", config.get("actors")),
        ("locations", config.get("locations")),
    ]:
        if value:
            parts.append(f"\n{label}: {json.dumps(value, ensure_ascii=False)}")
    return "\n".join(parts).strip()


_looks_like_screenplay_dump = looks_like_screenplay
