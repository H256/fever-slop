"""Compatibility imports for canonical project configuration validation."""

from feverslop.config.project_validation import (
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
