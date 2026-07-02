from __future__ import annotations

import contextlib
import threading
import time
import uuid
import subprocess
from pathlib import Path
from typing import Any, Callable

from feverslop.adapters.pipeline_runner import RunPipelineAdapter
from feverslop.composition import pipeline_runner
from feverslop.studio.logging import render_log_lines
from feverslop.tools.reference_bible import build_arg_parser as build_reference_bible_arg_parser
from feverslop.tools.reference_bible import run as render_reference_bible


JobHandler = Callable[[Callable[[str], None]], Any]
STREAM_CAPTURE_LOCK = threading.Lock()

PIPELINE_ACTIONS = {
    "main-pipeline",
    "storyboard",
    "msr-references",
    "msr-enrich",
    "ltx-render-scenes",
    "final-concat",
    "full-pipeline",
    "full-auto",
}


PIPELINE_STEPS: dict[str, list[str]] = {
    "main-pipeline": ["Main pipeline"],
    "storyboard": ["Storyboard", "Storyboard page"],
    "msr-references": ["MSR references"],
    "msr-enrich": ["MSR reference enrichment", "MSR prompt enrichment"],
    "ltx-render-scenes": ["LTX render"],
    "final-concat": ["Final concat"],
    "full-pipeline": [
        "Main pipeline",
        "MSR references",
        "MSR enrichment",
        "Storyboard",
        "LTX render",
        "Final concat",
    ],
    "full-auto": [
        "Song brief",
        "ACE-Step audio rendering",
        "Project scaffold",
        "Video pipeline",
    ],
}

STEP_ALIASES: dict[str, list[str]] = {
    "ACE-Step audio rendering": ["ACE-Step audio", "Rendering ACE-Step audio", "Generated audio"],
    "Project scaffold": ["Creating FeverSlop project", "Project config"],
}


class JobRegistry:
    def __init__(self):
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start(
        self,
        project_id: str,
        action: str,
        handler: JobHandler,
        *,
        project_type: str = "standard_music_video",
        reject_if_project_active: bool = False,
    ) -> str:
        job_id = uuid.uuid4().hex
        now = time.time()
        with self._lock:
            if reject_if_project_active and self._has_active_locked(project_id):
                raise ValueError("Pipeline is already running for this project")
            self._jobs[job_id] = {
                "id": job_id,
                "project_id": project_id,
                "project_type": project_type,
                "action": action,
                "pipeline_type": action,
                "status": "queued",
                "progress": 0,
                "overall_progress": 0,
                "current_step": None,
                "steps": self._initial_steps(action),
                "logs": [],
                "recent_logs": [],
                "error": None,
                "result": None,
                "created_at": now,
                "started_at": None,
                "completed_at": None,
                "updated_at": now,
                "elapsed_seconds": 0.0,
                "eta_seconds": None,
            }
        thread = threading.Thread(target=self._run, args=(job_id, handler), daemon=True)
        thread.start()
        return job_id

    def has_active_pipeline(self, project_id: str) -> bool:
        with self._lock:
            return self._has_active_locked(project_id)

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
                job = self._jobs[job_id]
                for line in render_log_lines(message):
                    job["logs"].append(line)
                    self._advance_step_from_log(job, line)
                job["logs"] = job["logs"][-500:]
                job["recent_logs"] = job["logs"][-100:]
                self._refresh_runtime(job)

        self._update(job_id, status="running", started_at=time.time())
        try:
            result = handler(log)
        except Exception as exc:  # noqa: BLE001 - job boundary should capture all failures
            self._update(job_id, status="failed", progress=100, overall_progress=100, completed_at=time.time(), error=str(exc))
            self._finish_current_step(job_id, "failed")
            return
        self._finish_all_steps(job_id)
        self._update(job_id, status="succeeded", progress=100, overall_progress=100, completed_at=time.time(), result=str(result) if result is not None else None)

    def _update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.update(fields)
            self._refresh_runtime(job)

    @staticmethod
    def _initial_steps(action: str) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "status": "pending",
                "progress": None,
                "started_at": None,
                "completed_at": None,
                "elapsed_seconds": 0.0,
            }
            for name in PIPELINE_STEPS.get(action, [action])
        ]

    def _advance_step_from_log(self, job: dict[str, Any], message: str) -> None:
        text = str(message).lower()
        steps = job.get("steps") or []
        if not steps:
            return
        current = next((step for step in steps if step["status"] == "running"), None)
        if current is None:
            current = steps[0]
            current["status"] = "running"
            current["started_at"] = time.time()
        for index, step in enumerate(steps):
            step_name = step["name"]
            step_matches = step_name.lower() in text or any(alias.lower() in text for alias in STEP_ALIASES.get(step_name, []))
            if step_matches and step is not current:
                if current and current["status"] == "running":
                    current["status"] = "completed"
                    current["progress"] = 100
                    current["completed_at"] = time.time()
                step["status"] = "running"
                step["started_at"] = step["started_at"] or time.time()
                current = step
                break
            if text.startswith("finished") and current and current["name"].lower() in text:
                current["status"] = "completed"
                current["progress"] = 100
                current["completed_at"] = time.time()
                next_step = steps[index + 1] if index + 1 < len(steps) else None
                if next_step:
                    next_step["status"] = "running"
                    next_step["started_at"] = time.time()
                    current = next_step
        job["current_step"] = current["name"] if current and current["status"] == "running" else None
        completed = sum(1 for step in steps if step["status"] == "completed")
        job["overall_progress"] = int((completed / len(steps)) * 100) if steps else job.get("progress", 0)
        job["progress"] = job["overall_progress"]

    def _finish_current_step(self, job_id: str, status: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for step in job.get("steps", []):
                if step["status"] == "running":
                    step["status"] = status
                    step["completed_at"] = time.time()
                    break
            self._refresh_runtime(job)

    def _finish_all_steps(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            now = time.time()
            for step in job.get("steps", []):
                if step["status"] in {"pending", "running"}:
                    step["status"] = "completed"
                    step["progress"] = 100
                    step["started_at"] = step["started_at"] or now
                    step["completed_at"] = now
            job["current_step"] = None
            self._refresh_runtime(job)

    @staticmethod
    def _refresh_runtime(job: dict[str, Any]) -> None:
        now = time.time()
        start = job.get("started_at") or job.get("created_at") or now
        end = job.get("completed_at") or now
        job["elapsed_seconds"] = max(0.0, end - start)
        for step in job.get("steps", []):
            step_start = step.get("started_at")
            if step_start:
                step_end = step.get("completed_at") or now
                step["elapsed_seconds"] = max(0.0, step_end - step_start)
        job["updated_at"] = now

    def _has_active_locked(self, project_id: str) -> bool:
        return any(
            job["project_id"] == project_id and job["action"] in PIPELINE_ACTIONS and job["status"] in {"queued", "running"}
            for job in self._jobs.values()
        )


def build_pipeline_options(action: str, *, scenes: list[int] | None = None, pipeline_mode: str | None = None) -> dict[str, Any]:
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
    video_pipeline = _video_pipeline_for_mode(pipeline_mode)
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
        base["video_pipeline"] = video_pipeline
        base["skip_ltx"] = False
        if scenes:
            base["scenes"] = ",".join(str(scene) for scene in scenes)
    elif action == "final-concat":
        base["video_pipeline"] = video_pipeline
        base["skip_final_concat"] = False
    elif action == "full-pipeline":
        return {
            "skip_tests": True,
            "video_pipeline": video_pipeline,
            "skip_msr_reference_render": video_pipeline != "ltx_msr",
            "skip_msr_prompt_enrichment": video_pipeline != "ltx_msr",
        }
    else:
        raise ValueError(f"Unknown pipeline action: {action}")
    return base


def build_pipeline_handler(project_config_path: Path, action: str, *, scenes: list[int] | None = None, pipeline_mode: str | None = None) -> JobHandler:
    options = build_pipeline_options(action, scenes=scenes, pipeline_mode=pipeline_mode)
    adapter = RunPipelineAdapter(run_pipeline=pipeline_runner.run, build_arg_parser=pipeline_runner.build_arg_parser)

    def run(log: Callable[[str], None]) -> Any:
        log(f"Starting {action}")
        result = run_with_stream_logging(lambda: adapter.run(project_config_path=project_config_path, options=options), log)
        log(f"Finished {action}")
        return result

    return run


def run_with_stream_logging(fn: Callable[[], Any], log: Callable[[str], None]) -> Any:
    writer = _StreamLogWriter(log)
    # ponytail: process-wide stdout redirection needs a global lock; move to subprocess streaming if parallel pipelines matter.
    with STREAM_CAPTURE_LOCK:
        with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
            try:
                return fn()
            finally:
                writer.flush()


class _StreamLogWriter:
    encoding = "utf-8"

    def __init__(self, log: Callable[[str], None]):
        self.log = log
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += str(text)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._emit(self._buffer)
            self._buffer = ""

    def isatty(self) -> bool:
        return False

    def _emit(self, text: str) -> None:
        for line in render_log_lines(text):
            if "<rich." not in line:
                self.log(line)


def _video_pipeline_for_mode(pipeline_mode: str | None) -> str:
    if pipeline_mode in {None, "", "classic", "ltx_i2v"}:
        return "ltx_i2v"
    if pipeline_mode in {"msr", "ltx_msr"}:
        return "ltx_msr"
    raise ValueError("pipeline_mode must be classic or msr")


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
