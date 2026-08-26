from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def patch_movie_msr_workflow(
    *,
    template_path: Path | None = None,
) -> dict[str, Any]:
    from feverslop.adapters.movie_workflow import MovieWorkflowPatcher

    template_path = template_path or Path(__file__).resolve().parents[3] / "workflows" / "video" / "ltx_25" / "msr" / "msr_draft.json"
    if not template_path.exists():
        raise FileNotFoundError(f"Movie MSR workflow template not found: {template_path}")
    workflow = json.loads(template_path.read_text(encoding="utf-8-sig"))
    return MovieWorkflowPatcher().strip_audio_inputs(workflow)
