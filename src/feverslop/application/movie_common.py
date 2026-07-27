from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from feverslop.errors import FeverSlopValidationError
from feverslop.domain.screenplay import looks_like_screenplay
from feverslop.domain.slug_utils import slugify_project_name


@dataclass(frozen=True)
class MovieInput:
    name: str
    source_type: str
    story_text: str
    desired_length: float
    width: int = 1280
    height: int = 704
    mode: str = "scaffold"
    min_scene_duration: float = 4.0
    max_scene_duration: float = 20.0
    config: dict = field(default_factory=dict)


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


_SCREENPLAY_HEADING_RE = re.compile(r"\b(?:INT|EXT|INT/EXT)\.\s+", re.IGNORECASE)
_DIALOGUE_CUE_RE = re.compile(r"\b[A-Z][A-Z0-9 _'-]{1,30}:\s+\S")


def validate_movie_input(request: MovieInput) -> None:
    if request.source_type not in {"short_story", "screenplay"}:
        raise FeverSlopValidationError("source_type must be short_story or screenplay")
    if not request.name.strip():
        raise FeverSlopValidationError("Movie project name is required")
    if not slugify_project_name(request.name):
        raise FeverSlopValidationError("Movie project slug is empty after slugifying the name")
    if len(request.story_text.strip()) < 20:
        raise FeverSlopValidationError("Movie story input is too short")
    if float(request.desired_length) <= 0:
        raise FeverSlopValidationError("desired_length must be positive")
    if int(request.width) <= 0 or int(request.height) <= 0:
        raise FeverSlopValidationError("resolution width and height must be positive")
    if request.mode not in {"scaffold", "full_auto"}:
        raise FeverSlopValidationError("movie mode must be scaffold or full_auto")
    if request.source_type == "screenplay" and not looks_like_screenplay(request.story_text):
        raise FeverSlopValidationError("screenplay input must contain scene headings such as INT. or EXT.")


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


def _looks_like_screenplay_dump(text: str) -> bool:
    if len(text) > 300:
        return True
    if _SCREENPLAY_HEADING_RE.search(text):
        return True
    return bool(_DIALOGUE_CUE_RE.search(text))
