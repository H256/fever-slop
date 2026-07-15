from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from feverslop.domain.prepared_workflow import PreparedSceneWorkflow


class WorkflowBackendPort(Protocol):
    def validate_workflow(self, workflow_path: Path, required_titles: list[str]) -> None:
        """Validate a backend workflow before rendering."""


@dataclass(frozen=True)
class WorkflowMaterializationRequest:
    scene: dict[str, Any]
    prompt: str
    audio_file: Path | None
    render_plan_path: Path
    pipeline: str
    seed: int | None = None


class WorkflowMaterializerPort(Protocol):
    def prepare(self, request: WorkflowMaterializationRequest) -> PreparedSceneWorkflow:
        """Persist one fully resolved workflow without queueing it."""


class PreparedWorkflowRendererPort(Protocol):
    def render(self, prepared_workflow_path: str | Path) -> Path:
        """Verify and render a previously materialized workflow."""
