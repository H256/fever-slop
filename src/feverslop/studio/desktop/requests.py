from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping

from feverslop.studio.projects import ProjectCreateRequest


def project_create_request(payload: Mapping[str, Any]) -> ProjectCreateRequest:
    values = dict(payload)
    silent_mode = values.get("silent_mode", False)
    if not isinstance(silent_mode, bool):
        raise ValueError("silent_mode must be a boolean")
    allowed = {field.name for field in fields(ProjectCreateRequest)}
    return ProjectCreateRequest(**{key: value for key, value in values.items() if key in allowed})
