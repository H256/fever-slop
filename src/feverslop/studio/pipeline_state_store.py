from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable


_MAIN_PIPELINE_DOWNSTREAM_STAGES = frozenset({
    "full-pipeline",
    "main_pipeline",
    "relay_compact",
    "anchor_fix",
    "storyboard_frames",
    "storyboard_page",
    "msr-references",
    "msr_references",
    "msr_reference_sheets",
    "msr_prompt_enrich",
    "msr-enrich",
    "ingredients_sheets",
    "ltx_prepare_workflows",
    "ltx_render_scenes",
    "ltx-prepare-workflows",
    "ltx-render-scenes",
    "concat_video_only",
    "mux_original_audio",
    "diagnostic_scene_audio_concat",
    "facefix",
    "facefix_concat",
    "final-concat",
})


def record_successful_stages(completed: list[str], *, action: str, stages: list[str]) -> list[str]:
    if action == "main-pipeline":
        completed = [stage for stage in completed if stage not in _MAIN_PIPELINE_DOWNSTREAM_STAGES]
    for stage in stages:
        if stage not in completed:
            completed.append(stage)
    return completed


def reconcile_completed_stages(state: dict[str, Any]) -> list[str]:
    completed = [str(stage) for stage in state.get("completed_stages") or []]
    runs = state.get("runs")
    if not isinstance(runs, list):
        return completed
    for run in runs:
        if not isinstance(run, dict) or run.get("status") != "succeeded":
            continue
        stages = run.get("stages")
        if not isinstance(stages, list):
            continue
        completed = record_successful_stages(
            completed,
            action=str(run.get("action") or ""),
            stages=[str(stage) for stage in stages],
        )
    return completed


class PipelineStateStore:
    def __init__(self, project_root: Callable[[str], Path], read_json_file: Callable[[Path], Any]):
        self.project_root = project_root
        self.read_json_file = read_json_file

    def record_pipeline_run(self, project_id: str, *, action: str, stages: list[str], status: str) -> dict[str, Any]:
        root = self.project_root(project_id)
        path = root / ".studio" / "pipeline_state.json"
        state = self.read_json_file(path)
        if not isinstance(state, dict):
            state = {}
        completed = list(state.get("completed_stages") or [])
        if status == "succeeded":
            completed = record_successful_stages(completed, action=action, stages=stages)
        entry = {
            "action": action,
            "stages": stages,
            "status": status,
            "updated_at": time.time(),
        }
        state["completed_stages"] = completed
        state["last_run"] = entry
        state.setdefault("runs", []).append(entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return state
