"""Compatibility imports for the canonical project composition service."""

from feverslop.composition.project_repository import (
    ProjectRepository,
    build_movie_planner,
    movie_default_config,
    movie_default_config_from_metadata,
    movie_project_config,
)

__all__ = [
    "ProjectRepository",
    "build_movie_planner",
    "movie_default_config",
    "movie_default_config_from_metadata",
    "movie_project_config",
]
