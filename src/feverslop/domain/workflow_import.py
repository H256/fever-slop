"""Data contracts for per-project ComfyUI workflow import.

The inspector, validator, and store are generic; per-family differences are
expressed purely as a :class:`WorkflowFamilySpec` (data), so a new pipeline
family does not require a new inspector implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Hard limits protecting the deterministic inspector from pathological inputs.
MAX_WORKFLOW_BYTES = 8 * 1024 * 1024
MAX_NODES = 2_000
MAX_EDGES = 10_000
MAX_NESTING = 32
MAX_STRING_BYTES = 64 * 1024


@dataclass(frozen=True)
class WorkflowFamilySpec:
    """Declarative description of one pipeline family's workflow boundary."""

    pipeline: str
    core_class_types: tuple[str, ...]
    required_inputs: tuple[str, ...]
    output_class_types: tuple[str, ...]
    seed_class_type: str | None = None
    seed_input: str | None = None


@dataclass(frozen=True)
class NodeCandidate:
    node_id: str
    class_type: str
    title: str = ""
    terminal: bool = False
    output_types: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "class_type": self.class_type,
            "title": self.title,
            "terminal": self.terminal,
            "output_types": list(self.output_types),
        }


@dataclass(frozen=True)
class AnalysisIssue:
    code: str
    message: str
    node_id: str | None = None
    input_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.node_id is not None:
            result["node_id"] = self.node_id
        if self.input_name is not None:
            result["input_name"] = self.input_name
        return result


@dataclass(frozen=True)
class BoundaryMapping:
    """The application-owned boundary around an opaque workflow graph."""

    core_node_id: str
    core_class_type: str
    required_inputs: tuple[str, ...]
    output_node_id: str
    seed_node_id: str | None = None
    seed_input: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "core_node_id": self.core_node_id,
            "core_class_type": self.core_class_type,
            "required_inputs": list(self.required_inputs),
            "output_node_id": self.output_node_id,
            "seed_node_id": self.seed_node_id,
            "seed_input": self.seed_input,
        }


@dataclass(frozen=True)
class WorkflowAnalysis:
    compatibility: str  # auto_compatible | needs_confirmation | unsupported
    mapping: BoundaryMapping | None = None
    core_candidates: tuple[NodeCandidate, ...] = ()
    output_candidates: tuple[NodeCandidate, ...] = ()
    issues: tuple[AnalysisIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatibility": self.compatibility,
            "mapping": self.mapping.to_dict() if self.mapping else None,
            "core_candidates": [c.to_dict() for c in self.core_candidates],
            "output_candidates": [c.to_dict() for c in self.output_candidates],
            "issues": [i.to_dict() for i in self.issues],
        }


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    issues: tuple[AnalysisIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "issues": [i.to_dict() for i in self.issues]}


@dataclass(frozen=True)
class TestRunResult:
    success: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"success": self.success, "detail": self.detail}


@dataclass(frozen=True)
class ImportedProfile:
    """A persisted, immutable imported workflow profile record."""

    profile_id: str
    pipeline: str
    purpose: str
    workflow_sha256: str
    mapping: BoundaryMapping | None
    status: str  # draft | validated | tested | active | broken
    test_run: TestRunResult | None = None
    analysis: WorkflowAnalysis | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "pipeline": self.pipeline,
            "purpose": self.purpose,
            "workflow_sha256": self.workflow_sha256,
            "mapping": self.mapping.to_dict() if self.mapping else None,
            "status": self.status,
            "test_run": self.test_run.to_dict() if self.test_run else None,
            "analysis": self.analysis.to_dict() if self.analysis else None,
        }
