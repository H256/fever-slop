"""Compatibility imports for the canonical reference workspace service."""

from feverslop.composition.reference_workspace_service import (
    CommandError,
    CommandResult,
    FakeGenerationJobRegistry,
    GenerationCommand,
    ReferenceWorkspaceService,
    StaleState,
    _asset_to_dict,
    _assignment_from_dict,
    _assignment_to_dict,
)

__all__ = [
    "CommandError",
    "CommandResult",
    "FakeGenerationJobRegistry",
    "GenerationCommand",
    "ReferenceWorkspaceService",
    "StaleState",
    "_asset_to_dict",
    "_assignment_from_dict",
    "_assignment_to_dict",
]
