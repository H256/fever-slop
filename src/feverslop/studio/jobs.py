from __future__ import annotations

import threading
import time
import uuid
import subprocess
from pathlib import Path
from typing import Any, Callable

from feverslop.adapters.pipeline_runner import RunPipelineAdapter
from feverslop.composition import pipeline_runner
from feverslop.tools.reference_bible import build_arg_parser as build_reference_bible_arg_parser
from feverslop.tools.reference_bible import run as render_reference_bible


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


def build_reference_rerender_handler(project_config_path: Path, *, reference_kind: str, reference_id: str) -> JobHandler:
    def run(log: Callable[[str], None]) -> Any:
        log(f"Rerendering {reference_kind} reference {reference_id}")
        args = build_reference_bible_arg_parser().parse_args(
            [
                "--project-config",
                str(project_config_path),
                "--app-config",
                "app_config.json",
                "--hero-workflow",
                str(Path("workflows") / "image_t2i_startframe_krea_v1.json"),
                "--edit-workflow",
                str(Path("workflows") / "image_edit_flux2_klein_1ref_v1.json"),
                "--view-set",
                "msr",
                "--only-kind",
                reference_kind,
                "--only-id",
                reference_id,
            ]
        )
        manifests = render_reference_bible(args)
        log(f"Finished {reference_kind} reference {reference_id}")
        return ", ".join(str(path) for path in manifests)

    return run


def build_ffmpeg_recut_command(
    raw_clip_path: Path,
    output_clip_path: Path,
    *,
    raw_in_seconds: float,
    raw_out_seconds: float,
    exact: bool = False,
) -> list[str]:
    if exact:
        return [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_clip_path),
            "-ss",
            f"{raw_in_seconds:.3f}",
            "-to",
            f"{raw_out_seconds:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-avoid_negative_ts",
            "make_zero",
            str(output_clip_path),
        ]
    return [
        "ffmpeg",
        "-y",
        "-ss",
        f"{raw_in_seconds:.3f}",
        "-to",
        f"{raw_out_seconds:.3f}",
        "-i",
        str(raw_clip_path),
        "-c",
        "copy",
        str(output_clip_path),
    ]


def build_recut_scene_handler(
    raw_clip_path: Path,
    output_clip_path: Path,
    *,
    raw_in_seconds: float,
    raw_out_seconds: float,
    exact: bool = False,
) -> JobHandler:
    def run(log: Callable[[str], None]) -> Any:
        if raw_out_seconds <= raw_in_seconds:
            raise ValueError("raw_out_seconds must be greater than raw_in_seconds")
        output_clip_path.parent.mkdir(parents=True, exist_ok=True)
        command = build_ffmpeg_recut_command(
            raw_clip_path,
            output_clip_path,
            raw_in_seconds=raw_in_seconds,
            raw_out_seconds=raw_out_seconds,
            exact=exact,
        )
        log(" ".join(command))
        subprocess.run(command, check=True)
        return output_clip_path

    return run
