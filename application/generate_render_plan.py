from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GenerateRenderPlanRequest:
    project_config_path: Path
    app_config_path: Path = Path("app_config.json")
    concept_batch_size: int = 0
    render_storyboard: bool = False
    zimage_workflow_path: Path | None = None
