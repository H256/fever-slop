from __future__ import annotations

import contextlib
import copy
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
import json
import re
import threading
import time
import uuid
import subprocess
from pathlib import Path
from typing import Any, Callable

from feverslop.adapters.pipeline_runner import RunPipelineAdapter
from feverslop.adapters.api_observability import api_observability_context
from feverslop.adapters.project_visual_consistency import (
    ProjectReferenceManifestAdapter,
    validate_project_scene_artifacts,
)
from feverslop.application.visual_consistency_preflight import (
    VisualConsistencyPreflightResult,
    preflight_visual_consistency,
    resolve_preflight_workflow_profile,
)
from feverslop.config.app_config import AppConfig
from feverslop.composition import pipeline_runner
from feverslop.composition.pipeline_runner import PipelineStage
from feverslop.config.project_config import ProjectConfig
from feverslop.domain.visual_consistency import PreflightMode
from feverslop.ports.timeline_documents import AffectedArtifacts
from feverslop.studio.logging import render_log_lines
from feverslop.tools.reference_bible import build_arg_parser as build_reference_bible_arg_parser
from feverslop.tools.reference_bible import run as render_reference_bible


JobHandler = Callable[[Callable[[str], None]], Any]
STREAM_CAPTURE_LOCK = threading.Lock()


class VisualConsistencyJobPayload(dict[str, Any]):
    pass


class StructuredJobLog(str):
    pass


class VisualConsistencyValidationError(ValueError):
    def __init__(self, payload: VisualConsistencyJobPayload) -> None:
        super().__init__("Visual consistency preflight blocked rendering")
        self.payload = payload

PIPELINE_ACTIONS = {
    "anchor-fix",
    "relay-compact",
    "storyboard-frames",
    "storyboard-page",
    "msr-reference-sheets",
    "msr-prompt-enrich",
    "rebuild-plan",
    "rebuild-plan-timeline",
    "concat-video-only",
    "mux-original-audio",
    "main-pipeline",
    "storyboard",
    "msr-references",
    "msr-enrich",
    "ltx-prepare-workflows",
    "ltx-render-scenes",
    "final-concat",
    "full-pipeline",
    "full-auto",
    "movie-full-auto",
    "movie-references",
    "movie-render",
    "movie-final-concat",
}


PIPELINE_STEPS: dict[str, list[str]] = {
    "anchor-fix": ["anchor fix"],
    "relay-compact": ["relay compact"],
    "storyboard-frames": ["Storyboard frames"],
    "storyboard-page": ["Storyboard page"],
    "msr-reference-sheets": ["MSR reference sheets"],
    "msr-prompt-enrich": ["MSR prompt enrichment"],
    "rebuild-plan": ["MSR reference sheets", "MSR prompt enrichment"],
    "concat-video-only": ["Final concat video-only"],
    "mux-original-audio": ["Mux original audio"],
    "main-pipeline": ["Main pipeline"],
    "storyboard": ["Storyboard", "Storyboard page"],
    "msr-references": ["MSR references"],
    "msr-enrich": ["MSR reference enrichment", "MSR prompt enrichment"],
    "ltx-prepare-workflows": ["Prepare LTX workflows"],
    "ltx-render-scenes": ["Video render"],
    "final-concat": ["Final concat"],
    "full-pipeline": [
        "Main pipeline",
        "MSR references",
        "MSR enrichment",
        "Storyboard",
        "Video render",
        "Final concat",
    ],
    "full-auto": [
        "Song brief",
        "ACE-Step audio rendering",
        "Project scaffold",
        "Video pipeline",
    ],
    "movie-full-auto": [
        "Story-Arch",
        "Scene planning",
        "Movie references",
        "Krea2 visual consistency",
        "LTX MSR native-audio render",
        "Final movie",
    ],
    "movie-references": ["Movie references"],
    "movie-render": ["LTX MSR native-audio render", "Final movie"],
    "movie-final-concat": ["Final movie"],
}

STEP_ALIASES: dict[str, list[str]] = {
    "ACE-Step audio rendering": ["ACE-Step audio", "Rendering ACE-Step audio", "Generated audio"],
    "Project scaffold": ["Creating FeverSlop project", "Project config"],
    "Story-Arch": ["Story-Arch Complete"],
    "Scene planning": ["Render Plan Ready", "Planning Scenes"],
    "Krea2 visual consistency": ["Krea2"],
    "Movie references": ["Movie references", "Reference sheets"],
    "Video render": ["ltx-render-scenes"],
    "LTX MSR native-audio render": ["LTX MSR", "native audio"],
    "Final movie": ["Movie Complete"],
    "MSR enrichment": ["MSR reference sheets", "MSR prompt enrichment"],
    "Storyboard": ["Storyboard frames", "Storyboard page"],
    "Ingredients sheets": ["Ingredients scene sheets"],
    "Final concat": ["Final concat video-only", "Mux original audio"],
}


FULL_PIPELINE_STEPS_BY_MODE = {
    "classic": ["Main pipeline", "Storyboard", "Video render", "Final concat"],
    "msr": ["Main pipeline", "MSR references", "MSR enrichment", "Video render", "Final concat"],
    "ingredients": ["Main pipeline", "MSR enrichment", "Ingredients sheets", "Video render", "Final concat"],
    "minimax_h3_r2v": ["Main pipeline", "MSR references", "Video render", "Final concat"],
    "minimax_h3_t2v": ["Main pipeline", "Video render", "Final concat"],
}


class JobRegistry:
    def __init__(self, *, max_workers: int = 4):
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="feverslop-studio-job",
        )
        self._futures: dict[Future[Any], str] = {}
        self._closed = False

    def start(
        self,
        project_id: str,
        action: str,
        handler: JobHandler,
        *,
        project_type: str = "standard_music_video",
        reject_if_project_active: bool = False,
        pipeline_mode: str | None = None,
    ) -> str:
        job_id = uuid.uuid4().hex
        now = time.time()
        with self._lock:
            if self._closed:
                raise RuntimeError("Job registry is shut down")
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
                "steps": self._initial_steps(action, pipeline_mode=pipeline_mode),
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
            future = self._executor.submit(self._run, job_id, handler)
            self._futures[future] = job_id
            future.add_done_callback(self._finish_future)
        return job_id

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def __enter__(self) -> JobRegistry:
        return self

    def __exit__(self, *_args: object) -> None:
        self.shutdown()

    def _finish_future(self, future: Future[Any]) -> None:
        with self._lock:
            job_id = self._futures.pop(future, None)
            if job_id is None or not future.cancelled():
                return
            job = self._jobs[job_id]
            if job["status"] == "queued":
                job.update(
                    status="cancelled",
                    completed_at=time.time(),
                    error="Job cancelled before execution",
                )
                self._refresh_runtime(job)

    def has_active_pipeline(self, project_id: str) -> bool:
        with self._lock:
            return self._has_active_locked(project_id)

    def list(self, project_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            jobs = [
                copy.deepcopy(job)
                for job in self._jobs.values()
                if project_id is None or job["project_id"] == project_id
            ]
        return sorted(jobs, key=lambda job: job["created_at"], reverse=True)

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return copy.deepcopy(self._jobs[job_id])

    def add_rebuild_plan_timeline(self, project_dir: str | Path, affected: AffectedArtifacts, rebuild_id: str | None = None) -> dict:
        """Schedule a rebuild job for invalidated downstream timeline artifacts.

        Only rebuilds artifacts whose corresponding flag is True in *affected*.
        Rebuilds execute in dependency order: beat -> scene -> stage1 -> prompt -> render.
        One failure does not stop the others.
        """
        project_dir_str = str(Path(project_dir).resolve())
        rebuild_id = rebuild_id or uuid.uuid4().hex[:12]
        affected_dict = {
            "beat_json": affected.beat_json,
            "scene_srt": affected.scene_srt,
            "stage1_segments": affected.stage1_segments,
            "ltx_prompt": affected.ltx_prompt,
            "render_plan": affected.render_plan,
        }
        payload = {
            "project_dir": project_dir_str,
            "affected": affected_dict,
            "rebuild_id": rebuild_id,
            "timestamp": time.time(),
        }

        def handler(log: Callable[[str], None]) -> dict[str, Any]:
            return _run_rebuild_plan_timeline(project_dir_str, affected_dict, log)

        project_id = Path(project_dir_str).name
        job_id = self.start(project_id, "rebuild-plan-timeline", handler)
        job = self.get(job_id)
        # Flatten payload fields into top level for easy access.
        job.update(payload)
        job["payload"] = payload
        return job

    def _run(self, job_id: str, handler: JobHandler) -> None:
        def log(message: str) -> None:
            with self._lock:
                job = self._jobs[job_id]
                lines = (
                    [str(message)]
                    if isinstance(message, StructuredJobLog)
                    else render_log_lines(message)
                )
                for line in lines:
                    job["logs"].append(line)
                    self._advance_step_from_log(job, line)
                job["logs"] = job["logs"][-500:]
                job["recent_logs"] = job["logs"][-100:]
                self._refresh_runtime(job)

        with self._lock:
            project_id = self._jobs[job_id]["project_id"]
        self._update(job_id, status="running", started_at=time.time())
        try:
            with api_observability_context(job_id=job_id, project_id=project_id):
                result = handler(log)
        except VisualConsistencyValidationError as exc:
            log(f"ERROR: {exc}")
            self._update(
                job_id,
                status="failed",
                progress=100,
                overall_progress=100,
                completed_at=time.time(),
                error=str(exc),
                result=dict(exc.payload),
            )
            self._finish_current_step(job_id, "failed")
            return
        except Exception as exc:  # noqa: BLE001 - job boundary should capture all failures
            log(f"ERROR: {exc}")
            self._update(job_id, status="failed", progress=100, overall_progress=100, completed_at=time.time(), error=str(exc))
            self._finish_current_step(job_id, "failed")
            return
        self._finish_all_steps(job_id)
        serialized_result = (
            dict(result)
            if isinstance(result, VisualConsistencyJobPayload)
            else str(result) if result is not None else None
        )
        self._update(job_id, status="succeeded", progress=100, overall_progress=100, completed_at=time.time(), result=serialized_result)

    def _update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.update(fields)
            self._refresh_runtime(job)

    @staticmethod
    def _initial_steps(action: str, *, pipeline_mode: str | None = None) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "status": "pending",
                "progress": None,
                "started_at": None,
                "completed_at": None,
                "elapsed_seconds": 0.0,
            }
            for name in _pipeline_step_names(action, pipeline_mode=pipeline_mode)
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
        if text.startswith("finished") and current:
            current_matches = current["name"].lower() in text or any(
                alias.lower() in text
                for alias in STEP_ALIASES.get(current["name"], [])
            )
            if current_matches:
                current["status"] = "completed"
                current["progress"] = 100
                current["completed_at"] = time.time()
                current_index = steps.index(current)
                next_step = steps[current_index + 1] if current_index + 1 < len(steps) else None
                if next_step:
                    next_step["status"] = "running"
                    next_step["started_at"] = time.time()
                    current = next_step
        clip_progress = re.search(r"rendered clip\s+(\d+)\s*/\s*(\d+)", text)
        if clip_progress:
            completed = int(clip_progress.group(1))
            total = max(1, int(clip_progress.group(2)))
            render_step = next((step for step in steps if step["name"] == "LTX MSR native-audio render"), current)
            if render_step:
                if current and current is not render_step and current["status"] == "running":
                    current["status"] = "completed"
                    current["progress"] = 100
                    current["completed_at"] = time.time()
                render_step["status"] = "running"
                render_step["started_at"] = render_step["started_at"] or time.time()
                render_step["progress"] = min(100, int((completed / total) * 100))
                current = render_step
        scene_progress = re.search(r"rendered scene\s+(\d+)\s*/\s*(\d+)", text)
        if scene_progress:
            completed = int(scene_progress.group(1))
            total = max(1, int(scene_progress.group(2)))
            render_step = next((step for step in steps if step["name"] == "Video render"), current)
            if render_step:
                if current and current is not render_step and current["status"] == "running":
                    current["status"] = "completed"
                    current["progress"] = 100
                    current["completed_at"] = time.time()
                render_step["status"] = "running"
                render_step["started_at"] = render_step["started_at"] or time.time()
                render_step["progress"] = min(100, int((completed / total) * 100))
                current = render_step
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


# ---------------------------------------------------------------------------
# Rebuild-plan-timeline handler
# ---------------------------------------------------------------------------


def _run_rebuild_plan_timeline(project_dir: str, affected: dict[str, bool], log: Callable[[str], None]) -> dict[str, Any]:
    """Rebuild downstream artifacts via the pipeline, skipping audio analysis.

    Calls execute_rebuild_render_plan which assembles SceneTimelinePipeline,
    PromptGenerationPipeline, and RenderPlanPipeline to regenerate all
    affected artifacts from the current timeline on disk.
    """
    from feverslop.application.generate_render_plan import (
        GenerateRenderPlanRequest,
    )
    from feverslop.composition.generate_render_plan import (
        execute_rebuild_render_plan,
    )

    log("Rebuilding render plan from timeline data...")
    config_path = Path(project_dir) / "config.json"
    try:
        request = GenerateRenderPlanRequest(project_config_path=config_path)
        result = execute_rebuild_render_plan(request)
        log(f"  Render plan rebuilt: {result.scene_count} scenes")
        return {
            "total": len(affected),
            "rebuilt": len(affected),
            "failed": 0,
            "result": result,
        }
    except Exception as exc:
        log(f"  Rebuild failed: {exc}")
        return {
            "total": len(affected),
            "rebuilt": 0,
            "failed": 1,
            "error": str(exc),
        }


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
    if action == "anchor-fix":
        base["stages"] = [PipelineStage.ANCHOR_FIX.value]
    elif action == "relay-compact":
        base["render_mode"] = "relay"
        base["stages"] = [PipelineStage.RELAY_COMPACT.value]
    elif action == "storyboard-frames":
        base["stages"] = [PipelineStage.STORYBOARD_FRAMES.value]
    elif action == "storyboard-page":
        base["stages"] = [PipelineStage.STORYBOARD_PAGE.value]
    elif action == "msr-reference-sheets":
        base["video_pipeline"] = "ltx_msr"
        base["stages"] = [PipelineStage.MSR_REFERENCE_SHEETS.value]
    elif action == "msr-prompt-enrich":
        base["video_pipeline"] = "ltx_msr"
        base["stages"] = [PipelineStage.MSR_PROMPT_ENRICH.value]
    elif action == "rebuild-plan":
        base["video_pipeline"] = "ltx_msr"
        base["stages"] = [PipelineStage.MSR_REFERENCE_SHEETS.value, PipelineStage.MSR_PROMPT_ENRICH.value]
        base["skip_msr_prompt_enrichment"] = False
    elif action == "concat-video-only":
        base["video_pipeline"] = video_pipeline
        base["stages"] = [PipelineStage.CONCAT_VIDEO_ONLY.value]
    elif action == "mux-original-audio":
        base["video_pipeline"] = video_pipeline
        base["stages"] = [PipelineStage.MUX_ORIGINAL_AUDIO.value]
    elif action == "main-pipeline":
        base["skip_main_pipeline"] = False
    elif action == "storyboard":
        base["stages"] = [PipelineStage.STORYBOARD_FRAMES.value, PipelineStage.STORYBOARD_PAGE.value]
        base["skip_storyboard"] = False
        base["skip_storyboard_page"] = False
    elif action == "msr-references":
        base["video_pipeline"] = "ltx_msr"
        base["skip_msr_reference_render"] = False
    elif action == "msr-enrich":
        base["video_pipeline"] = "ltx_msr"
        base["stages"] = [PipelineStage.MSR_REFERENCE_SHEETS.value, PipelineStage.MSR_PROMPT_ENRICH.value]
        base["skip_msr_prompt_enrichment"] = False
    elif action == "ingredients-sheets":
        base["video_pipeline"] = "ltx_ingredients"
        base["stages"] = [PipelineStage.INGREDIENTS_SHEETS.value]
    elif action == "ltx-prepare-workflows":
        base["video_pipeline"] = video_pipeline
        base["stages"] = [PipelineStage.LTX_PREPARE_WORKFLOWS.value]
        base["skip_ltx"] = False
        if scenes:
            base["scenes"] = ",".join(str(scene) for scene in scenes)
    elif action == "ltx-render-scenes":
        base["video_pipeline"] = video_pipeline
        base["stages"] = [PipelineStage.LTX_RENDER_SCENES.value]
        base["skip_ltx"] = False
        if scenes:
            base["scenes"] = ",".join(str(scene) for scene in scenes)
    elif action == "final-concat":
        base["video_pipeline"] = video_pipeline
        base["stages"] = [PipelineStage.CONCAT_VIDEO_ONLY.value, PipelineStage.MUX_ORIGINAL_AUDIO.value]
        base["skip_final_concat"] = False
    elif action == "full-pipeline":
        return {
            "skip_tests": True,
            "video_pipeline": video_pipeline,
            "skip_msr_reference_render": video_pipeline not in {"ltx_msr", "ltx_ingredients", "minimax-h3-r2v"},
            "skip_msr_prompt_enrichment": video_pipeline not in {"ltx_msr", "ltx_ingredients"},
            "skip_ingredients_sheets": video_pipeline != "ltx_ingredients",
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


def build_visual_consistency_preflight_handler(
    project_dir: Path,
    *,
    plan_path: Path,
    mode: str,
    preflight_mode: PreflightMode,
    workflow_profile: str | None = None,
) -> JobHandler:
    project_dir = project_dir.resolve()
    plan_path = plan_path.resolve()
    if not plan_path.is_relative_to(project_dir):
        raise ValueError("Visual consistency plan must be inside the project")

    def run(log: Callable[[str], None]) -> VisualConsistencyJobPayload:
        payload = json.loads(plan_path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, list):
            scenes = payload
        elif isinstance(payload, dict):
            scenes = payload.get("scenes", payload.get("shots"))
        else:
            scenes = None
        if not isinstance(scenes, list) or not all(
            isinstance(scene, dict) for scene in scenes
        ):
            raise ValueError(
                "Render plan must be a JSON array or contain scenes/shots"
            )
        if preflight_mode is PreflightMode.OFF:
            result = VisualConsistencyPreflightResult((), ())
        else:
            config = ProjectConfig.load(project_dir / "config.json")
            snapshot = ProjectReferenceManifestAdapter(
                lambda _project_id: project_dir
            ).load(project_dir.name)
            app_config = AppConfig.load("app_config.json")
            pipeline = {
                "ingredients": "ltx_ingredients",
                "msr": "ltx_msr",
                "i2v": "ltx_i2v",
            }[mode]
            configured_profile = app_config.resolve_video_workflow_profile(
                pipeline=pipeline,
                purpose="final",
            )
            resolved_profile = resolve_preflight_workflow_profile(
                scenes,
                explicit_profile=workflow_profile,
                legacy_fallback=(
                    configured_profile.name
                    if configured_profile is not None
                    else f"{mode}-default"
                ),
            )
            selected_profile = next(
                (
                    profile
                    for profile in app_config.video_workflow_profiles
                    if profile.name == resolved_profile
                    and profile.pipeline == pipeline
                    and profile.purpose == "final"
                ),
                None,
            )
            result = preflight_visual_consistency(
                scenes,
                snapshot,
                mode=mode,
                workflow_profile=resolved_profile,
                preflight_mode=preflight_mode,
                subject_mode=config.subject_mode,
                max_scene_actors=config.max_scene_actors,
                supports_continuous_transitions=(
                    mode != "ingredients"
                    and selected_profile is not None
                    and selected_profile.supports_start_frame
                ),
            )
            artifact_issues = validate_project_scene_artifacts(
                project_dir,
                scenes,
                mode=mode,
                preflight_mode=preflight_mode,
            )
            result = VisualConsistencyPreflightResult(
                result.contracts,
                (*result.issues, *artifact_issues),
            )
        log(
            f"Visual consistency preflight: "
            f"{'renderable' if result.renderable else 'blocked'}; "
            f"{len(result.issues)} issue(s)"
        )
        for issue in result.issues:
            log(
                f"{issue.severity.upper()} scene {issue.scene} "
                f"{issue.code}: {issue.message}"
            )
        payload = VisualConsistencyJobPayload(
            renderable=result.renderable,
            contracts=[contract.to_dict() for contract in result.contracts],
            issues=[asdict(issue) for issue in result.issues],
        )
        log(StructuredJobLog(json.dumps(payload, sort_keys=True)))
        if not result.renderable:
            raise VisualConsistencyValidationError(payload)
        return payload

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
    if pipeline_mode in {"ingredients", "ltx_ingredients"}:
        return "ltx_ingredients"
    if pipeline_mode in {"minimax_h3_r2v", "minimax-h3-r2v"}:
        return "minimax-h3-r2v"
    if pipeline_mode in {"minimax_h3_t2v", "minimax-h3-t2v"}:
        return "minimax-h3-t2v"
    raise ValueError("pipeline_mode must be classic, msr, ingredients, minimax_h3_r2v, or minimax_h3_t2v")


def _pipeline_step_names(action: str, *, pipeline_mode: str | None = None) -> list[str]:
    if action != "full-pipeline":
        return PIPELINE_STEPS.get(action, [action])
    if pipeline_mode in {"msr", "ltx_msr"}:
        return FULL_PIPELINE_STEPS_BY_MODE["msr"]
    if pipeline_mode in {"ingredients", "ltx_ingredients"}:
        return FULL_PIPELINE_STEPS_BY_MODE["ingredients"]
    if pipeline_mode in {"minimax_h3_r2v", "minimax-h3-r2v"}:
        return FULL_PIPELINE_STEPS_BY_MODE["minimax_h3_r2v"]
    if pipeline_mode in {"minimax_h3_t2v", "minimax-h3-t2v"}:
        return FULL_PIPELINE_STEPS_BY_MODE["minimax_h3_t2v"]
    return FULL_PIPELINE_STEPS_BY_MODE["classic"]


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
