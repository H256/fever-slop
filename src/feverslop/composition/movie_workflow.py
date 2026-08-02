from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def patch_movie_msr_workflow(
    *,
    template_path: Path = Path("workflows") / "video_default_ltxv_msr_1actor_1background_v4.json",
) -> dict[str, Any]:
    from feverslop.adapters.movie_workflow import MovieWorkflowPatcher

    if not template_path.exists():
        raise FileNotFoundError(f"Movie MSR workflow template not found: {template_path}")
    workflow = json.loads(template_path.read_text(encoding="utf-8"))
    return MovieWorkflowPatcher().strip_audio_inputs(workflow)
