from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable


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
            for stage in stages:
                if stage not in completed:
                    completed.append(stage)
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
