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

from pathlib import Path
from typing import Any

from feverslop.domain.render_profile import PostprocessStrategy, QualityProfile, RenderPassStrategy
from feverslop.path_utils import resolve_workflow_reference
from feverslop.ports.project_requests import ProjectCreateRequest

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


def movie_project_config(request: ProjectCreateRequest) -> dict[str, Any]:
    """Build the validated movie-specific configuration for a new project."""
    planner_backend = _movie_planner_backend(request.movie_planner_backend)
    reference_backend = _supported_backend(request.movie_reference_backend, "movie_reference_backend", {"comfyui", "local"}, default="comfyui")
    render_backend = _supported_backend(request.movie_render_backend, "movie_render_backend", {"comfyui", "local"}, default="comfyui")
    movie_video_workflow = _supported_backend(request.movie_video_workflow, "movie_video_workflow", {"msr", "msr-i2v-startframe", "i2v-edit", "startframe-director", "ingredients"}, default="msr")
    startframe_director_backend = _supported_backend(request.movie_startframe_director_backend, "movie_startframe_director_backend", {"krea2", "ideogram"}, default="krea2")
    continuity_keyframes = _supported_backend(request.movie_continuity_keyframes, "movie_continuity_keyframes", {"none", "last-to-start"}, default="none")
    if continuity_keyframes == "last-to-start" and movie_video_workflow != "msr-i2v-startframe":
        raise ValueError("movie_continuity_keyframes=last-to-start requires movie_video_workflow=msr-i2v-startframe")
    return {
        "planner_backend": planner_backend, "reference_backend": reference_backend,
        "render_backend": render_backend, "movie_video_workflow": movie_video_workflow,
        "startframe_director_backend": startframe_director_backend,
        "continuity_keyframes": continuity_keyframes,
        "dialogue_language": _dialogue_language(request.dialogue_language),
        "hero_workflow": _project_workflow_path(request.movie_hero_workflow, "movie_hero_workflow"),
        "edit_workflow": _movie_edit_workflow(request.movie_edit_workflow, movie_video_workflow=movie_video_workflow),
        "director_workflow": _project_workflow_path(_default_movie_director_workflow(request.movie_director_workflow, startframe_director_backend), "movie_director_workflow"),
        "mask_workflow": _project_workflow_path(request.movie_mask_workflow, "movie_mask_workflow"),
        "identity_repair_workflow": _project_workflow_path(request.movie_identity_repair_workflow, "movie_identity_repair_workflow"),
        "detail_workflow": _project_workflow_path(request.movie_detail_workflow, "movie_detail_workflow"),
        "startframe_comfyui_base_url": str(request.movie_startframe_comfyui_base_url or "http://localhost:8188").rstrip("/"),
        "startframe_write_debug_workflows": bool(request.movie_startframe_write_debug_workflows),
        "startframe_debug_workflows_dir": _project_optional_relative_path(request.movie_startframe_debug_workflows_dir, "movie_startframe_debug_workflows_dir"),
        "startframe_validator_base_url": str(request.movie_startframe_validator_base_url or "http://your-llm-server.local/v1").rstrip("/"),
        "startframe_validator_model": str(request.movie_startframe_validator_model or "gemma4-26b-a4b:vision"),
        "msr_workflow": _project_workflow_path(request.movie_msr_workflow, "movie_msr_workflow"),
        "msr_i2v_workflow": _project_workflow_path(request.movie_msr_i2v_workflow, "movie_msr_i2v_workflow"),
        "i2v_workflow": _project_workflow_path(request.movie_i2v_workflow, "movie_i2v_workflow"),
        "refine_location_prompts": bool(request.movie_refine_location_prompts),
        "refine_actor_prompts": bool(request.movie_refine_actor_prompts),
        "render_profile": _render_profile_config(request),
    }


def _movie_planner_backend(value: str) -> str:
    normalized = str(value or "llm").strip().lower()
    if normalized in {"local", "placeholder"}:
        normalized = "deterministic"
    if normalized not in {"llm", "deterministic"}:
        raise ValueError("movie_planner_backend must be llm or deterministic")
    return normalized


def _render_profile_config(request: ProjectCreateRequest) -> dict[str, str]:
    try:
        return {"quality": QualityProfile(str(request.render_quality).strip().lower()).value, "pass_strategy": RenderPassStrategy(str(request.render_pass_strategy).strip().lower()).value, "postprocess": PostprocessStrategy(str(request.render_postprocess).strip().lower()).value}
    except ValueError as exc:
        raise ValueError("render profile requires quality draft/standard/final, pass strategy single_pass/two_pass, and postprocess none/seedvr") from exc


def _supported_backend(value: str, field: str, supported: set[str], *, default: str) -> str:
    normalized = str(value or "").strip().lower() or default
    if normalized == "placeholder":
        normalized = "local"
    if normalized not in supported:
        raise ValueError(f"{field} must be one of: {', '.join(sorted(supported))}")
    return normalized


def _project_workflow_path(value: str, field: str) -> str:
    path = resolve_workflow_reference(str(value or "").strip())
    if not path:
        raise ValueError(f"{field} is required")
    parsed = Path(path)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise ValueError(f"{field} must be a repository-relative path")
    return parsed.as_posix()


def _project_optional_relative_path(value: str, field: str) -> str:
    parsed = Path(str(value or "").strip())
    if not str(value or "").strip():
        return ""
    if parsed.is_absolute() or ".." in parsed.parts:
        raise ValueError(f"{field} must be a repository-relative path")
    return parsed.as_posix()


def _movie_edit_workflow(value: str, *, movie_video_workflow: str) -> str:
    default = "workflows/image/image-model/image_edit_flux2_klein_1ref_v1.json"
    if movie_video_workflow == "i2v-edit" and str(value or "").strip() in {"", default}:
        value = "workflows/image/image-model/image_edit_flux2_klein_2ref_v1.json"
    return _project_workflow_path(value, "movie_edit_workflow")


def _default_movie_director_workflow(value: str, backend: str) -> str:
    default = "workflows/image/image-model/image_t2i_startframe_krea_v1.json"
    if value and backend == "ideogram" and value == default:
        return "workflows/image/image-model/image_t2i_startframe_ideogram_director_v1.json"
    return value or ("workflows/image/image-model/image_t2i_startframe_ideogram_director_v1.json" if backend == "ideogram" else default)


def _dialogue_language(value: object) -> str:
    return str(value or "").strip() or "English"

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
