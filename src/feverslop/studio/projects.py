"""Compatibility imports for the canonical project store composition service."""

import logging
import os

from feverslop.composition import project_store as _canonical_project_store

from feverslop.composition.project_store import (
    AUDIO_EXTENSIONS,
    AUDIO_MIME_TYPES,
    ArtifactCatalog,
    ArtifactConflict,
    ArtifactRequest,
    ProjectCreateRequest,
    ProjectStore,
    RenderPlanPatch,
    StudioPathError,
    _artifact_revision,  # noqa: F401
    _path_revision,  # noqa: F401
    sanitize_audio_filename,
    slugify_project_name,
)

_canonical_project_store._logger = logging.getLogger(__name__)
_canonical_project_store.os = os

__all__ = [
    "AUDIO_EXTENSIONS",
    "AUDIO_MIME_TYPES",
    "ArtifactCatalog",
    "ArtifactConflict",
    "ArtifactRequest",
    "ProjectCreateRequest",
    "ProjectStore",
    "RenderPlanPatch",
    "StudioPathError",
    "sanitize_audio_filename",
    "slugify_project_name",
]
