from __future__ import annotations

from pathlib import Path
from typing import Protocol


class WorkflowBackendPort(Protocol):
    def validate_workflow(self, workflow_path: Path, required_titles: list[str]) -> None:
        """Validate a backend workflow before rendering."""
