"""Re-export module for backwards compatibility.

All movie-related application logic has been split into focused modules.
Import from the new modules directly for clarity:
  - feverslop.application.movie_common
  - feverslop.application.movie_use_cases
  - feverslop.application.movie_bible
  - feverslop.application.movie_continuity
  - feverslop.application.movie_references
"""
from __future__ import annotations

# Bible generation and normalization
from feverslop.application.movie_bible import (
    _bible_dict,
    _reference_manifest,
    _render_plan,
    augment_movie_bible_from_shot_references,
    constrain_movie_shots_to_bible,
    generate_movie_bible,
    generate_movie_continuity_plan,
    movie_bible_from_dict,
    plan_movie_shots_from_bible,
)

# Shared types and helpers
from feverslop.application.movie_common import (
    MovieInput,
    MovieProductionResult,
    MovieScaffoldResult,
    _looks_like_screenplay_dump,
    _planner_source_text,
)

# Continuity planning
from feverslop.application.movie_continuity import (
    apply_movie_continuity_to_shots,
    build_movie_continuity_fallback,
    movie_continuity_plan_from_dict,
    movie_continuity_plan_to_dict,
)

# Reference prompts
from feverslop.application.movie_references import (
    build_movie_actor_reference_prompt,
    build_movie_actor_visual_description,
)

# Use cases
from feverslop.application.movie_use_cases import (
    AutoProduceMovieUseCase,
    ScaffoldMovieUseCase,
)

# Domain types still needed by callers
from feverslop.domain.movie import CinematicShot, MovieShotCard

# Backwards-compatible re-exports of domain utilities (underscore-prefixed)
from feverslop.domain.movie_utils import safe_id as _safe_id
from feverslop.domain.movie_utils import string_list as _string_list

# Slug utility
from feverslop.domain.slug_utils import slugify_project_name

__all__ = [
    # Types
    "CinematicShot",
    "MovieInput",
    "MovieProductionResult",
    "MovieScaffoldResult",
    "MovieShotCard",
    # Use cases
    "AutoProduceMovieUseCase",
    "ScaffoldMovieUseCase",
    # Bible
    "_bible_dict",
    "_reference_manifest",
    "_render_plan",
    "augment_movie_bible_from_shot_references",
    "constrain_movie_shots_to_bible",
    "generate_movie_bible",
    "generate_movie_continuity_plan",
    "movie_bible_from_dict",
    "plan_movie_shots_from_bible",
    # Continuity
    "apply_movie_continuity_to_shots",
    "build_movie_continuity_fallback",
    "movie_continuity_plan_from_dict",
    "movie_continuity_plan_to_dict",
    # References
    "build_movie_actor_reference_prompt",
    "build_movie_actor_visual_description",
    # Helpers
    "_looks_like_screenplay_dump",
    "_planner_source_text",
    "_safe_id",
    "_string_list",
    "slugify_project_name",
]
