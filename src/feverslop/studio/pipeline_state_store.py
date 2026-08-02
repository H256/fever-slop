from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable


_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[Path, threading.RLock] = {}


def _lock_for_path(path: Path) -> threading.RLock:
    canonical_path = path.resolve()
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(canonical_path, threading.RLock())


class PipelineStateStore:
    def __init__(self, project_root: Callable[[str], Path], read_json_file: Callable[[Path], Any]):
        self.project_root = project_root
        self.read_json_file = read_json_file

    def record_pipeline_run(self, project_id: str, *, action: str, stages: list[str], status: str) -> dict[str, Any]:
        root = self.project_root(project_id)
        path = root / ".studio" / "pipeline_state.json"
        with _lock_for_path(path):
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
            temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                temporary_path.write_text(
                    json.dumps(state, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                temporary_path.replace(path)
            finally:
                temporary_path.unlink(missing_ok=True)
            return state
