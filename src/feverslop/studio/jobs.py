from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from feverslop.adapters.pipeline_runner import RunPipelineAdapter
from feverslop.composition import pipeline_runner


JobHandler = Callable[[Callable[[str], None]], Any]


class JobRegistry:
    def __init__(self):
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start(self, project_id: str, action: str, handler: JobHandler) -> str:
        job_id = uuid.uuid4().hex
        now = time.time()
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id,
                "project_id": project_id,
                "action": action,
                "status": "queued",
                "progress": 0,
                "logs": [],
                "error": None,
                "result": None,
                "created_at": now,
                "updated_at": now,
            }
        thread = threading.Thread(target=self._run, args=(job_id, handler), daemon=True)
        thread.start()
        return job_id

    def list(self, project_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            jobs = [dict(job) for job in self._jobs.values() if project_id is None or job["project_id"] == project_id]
        return sorted(jobs, key=lambda job: job["created_at"], reverse=True)

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return dict(self._jobs[job_id])

    def _run(self, job_id: str, handler: JobHandler) -> None:
        def log(message: str) -> None:
            with self._lock:
                self._jobs[job_id]["logs"].append(message)
                self._jobs[job_id]["updated_at"] = time.time()

        self._update(job_id, status="running", progress=5)
        try:
            result = handler(log)
        except Exception as exc:  # noqa: BLE001 - job boundary should capture all failures
            self._update(job_id, status="failed", progress=100, error=str(exc))
            return
        self._update(job_id, status="succeeded", progress=100, result=str(result) if result is not None else None)

    def _update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(fields)
            self._jobs[job_id]["updated_at"] = time.time()


def build_pipeline_options(action: str, *, scenes: list[int] | None = None) -> dict[str, Any]:
    base: dict[str, Any] = {
        "skip_tests": True,
        "skip_main_pipeline": True,
        "skip_relay_compact": True,
        "skip_anchor_fix": True,
        "skip_storyboard": True,
        "skip_storyboard_page": True,
        "skip_msr_reference_render": True,
        "skip_msr_prompt_enrichment": True,
        "skip_ltx": True,
        "skip_final_concat": True,
    }
    if action == "main-pipeline":
        base["skip_main_pipeline"] = False
    elif action == "storyboard":
        base["skip_storyboard"] = False
        base["skip_storyboard_page"] = False
    elif action == "msr-references":
        base["video_pipeline"] = "ltx_msr"
        base["skip_msr_reference_render"] = False
    elif action == "msr-enrich":
        base["video_pipeline"] = "ltx_msr"
        base["skip_msr_prompt_enrichment"] = False
    elif action == "ltx-render-scenes":
        base["video_pipeline"] = "ltx_msr"
        base["skip_ltx"] = False
        if scenes:
            base["scenes"] = ",".join(str(scene) for scene in scenes)
    elif action == "final-concat":
        base["video_pipeline"] = "ltx_msr"
        base["skip_final_concat"] = False
    elif action == "full-pipeline":
        return {"skip_tests": True}
    else:
        raise ValueError(f"Unknown pipeline action: {action}")
    return base


def build_pipeline_handler(project_config_path: Path, action: str, *, scenes: list[int] | None = None) -> JobHandler:
    options = build_pipeline_options(action, scenes=scenes)
    adapter = RunPipelineAdapter(run_pipeline=pipeline_runner.run, build_arg_parser=pipeline_runner.build_arg_parser)

    def run(log: Callable[[str], None]) -> Any:
        log(f"Starting {action}")
        result = adapter.run(project_config_path=project_config_path, options=options)
        log(f"Finished {action}")
        return result

    return run

