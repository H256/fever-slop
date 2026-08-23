"""Project persistence contracts and transport request mapping."""

from feverslop.ports.project_requests import (
    AUDIO_EXTENSIONS,
    AUDIO_MIME_TYPES,
    ArtifactConflict,
    ArtifactRequest,
    ProjectCreateRequest,
    RenderPlanPatch,
    StudioPathError,
    sanitize_audio_filename,
)

from collections.abc import Mapping
from dataclasses import fields
from typing import Any


def project_create_request(payload: Mapping[str, Any]) -> ProjectCreateRequest:
    """Build a project request while ignoring unknown transport fields."""
    values = dict(payload)
    silent_mode = values.get("silent_mode", False)
    if not isinstance(silent_mode, bool):
        raise ValueError("silent_mode must be a boolean")
    allowed = {field.name for field in fields(ProjectCreateRequest)}
    return ProjectCreateRequest(**{key: value for key, value in values.items() if key in allowed})

__all__ = [
    "AUDIO_EXTENSIONS",
    "AUDIO_MIME_TYPES",
    "ArtifactConflict",
    "ArtifactRequest",
    "ProjectCreateRequest",
    "RenderPlanPatch",
    "StudioPathError",
    "sanitize_audio_filename",
    "project_create_request",
]
