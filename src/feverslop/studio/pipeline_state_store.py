"""Compatibility imports for the canonical pipeline-state adapter."""

from feverslop.adapters.pipeline_state_store import (
    _MAIN_PIPELINE_DOWNSTREAM_STAGES,  # noqa: F401 - legacy private import compatibility
    _PATH_LOCKS,  # noqa: F401 - legacy private import compatibility
    _lock_for_path,  # noqa: F401 - legacy private import compatibility
    PipelineStateStore,
    reconcile_completed_stages,
    record_successful_stages,
)

__all__ = [
    "PipelineStateStore",
    "reconcile_completed_stages",
    "record_successful_stages",
]
