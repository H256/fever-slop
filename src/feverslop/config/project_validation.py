"""Compatibility imports for the project configuration registry."""

from feverslop.config.project_config import (
    VIDEO_PIPELINE_BY_MODE,
    validate_full_auto_inputs,
    validate_pipeline_mode,
    validate_project_config,
)

__all__ = [
    "VIDEO_PIPELINE_BY_MODE",
    "validate_full_auto_inputs",
    "validate_pipeline_mode",
    "validate_project_config",
]
