from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from feverslop.application.job_contracts import JobRequest
from feverslop.composition.movie_pipeline_jobs import (  # noqa: F401
    backend_config_path,
    build_movie_full_auto_handler,
    build_movie_i2v_edit_visual_adapter,
    build_movie_reference_generator,
    build_movie_references_handler,
    build_movie_render_handler,
    build_movie_startframe_director_visual_adapter,
    build_movie_visual_adapter,
    ensure_movie_references,
    mark_movie_reference_backend,
    movie_config_from_metadata,
    movie_references_ready,
    movie_runtime_config,
    patch_movie_msr_workflow,
    sync_movie_manifest_with_render_plan,
    write_startframe_i2v_empty_audio_workflow,
)
from feverslop.ports.rebuild_execution import ArtifactProvenancePort
from feverslop.composition.job_runtime import (
    PIPELINE_ACTIONS,
    JobHandler,
    JobRegistry,
    build_pipeline_handler,
    build_pipeline_options,
    build_recut_scene_handler,
    build_reference_rerender_handler,
    build_visual_consistency_preflight_handler,
    run_with_stream_logging,
)
from feverslop.composition.logging import render_log_lines

from feverslop.composition.pipeline_actions import ensure_pipeline_action_available
from feverslop.config.project_validation import VIDEO_PIPELINE_BY_MODE
from feverslop.studio.projects import ProjectStore

FullAutoHandlerFactory = Callable[..., Any]
PipelineHandlerFactory = Callable[..., Any]


StudioJobRequest = JobRequest


class ActionHandler(Protocol):
    action: str

    def build(self, project_id: str, request: StudioJobRequest, metadata: dict[str, Any]) -> JobHandler:
        """Build a runnable job handler."""


class ActionRegistry:
    def __init__(self, handlers: list[ActionHandler], fallback: ActionHandler):
        self._handlers = {handler.action: handler for handler in handlers}
        self._fallback = fallback

    def resolve(self, action: str) -> ActionHandler:
        return self._handlers.get(action, self._fallback)


class StudioJobService:
    def __init__(
        self,
        *,
        store: ProjectStore,
        jobs: JobRegistry,
        provenance: ArtifactProvenancePort | None = None,
        full_auto_handler: FullAutoHandlerFactory | None = None,
        pipeline_handler: PipelineHandlerFactory | None = None,
    ):
        self.store = store
        self.jobs = jobs
        self.provenance = provenance
        self.registry = ActionRegistry(
            [
                ReferenceRerenderAction(store),
                RecutSceneAction(store),
                ThumbnailPrebuildAction(store),
                ThumbnailCleanupAction(store),
                FullAutoAction(store, full_auto_handler),
                MovieReferencesAction(store),
                MovieRenderAction(store),
                MovieFinalConcatAction(store),
                MovieFullAutoAction(store),
                VisualConsistencyPreflightAction(store),
            ],
            PipelineAction(store, pipeline_handler, provenance),
        )

    def start_job(self, project_id: str, request: StudioJobRequest) -> dict[str, Any]:
        metadata = self.store.project_metadata(project_id)
        handler = self.registry.resolve(request.action).build(project_id, request, metadata)
        pipeline_mode = request.pipeline_mode
        if request.action in PIPELINE_ACTIONS and not pipeline_mode and metadata.get("project_type") != "movie":
            pipeline_mode = pipeline_mode_from_config(self.store.resolve_project_path(project_id, "config.json"))
        return self.jobs.get(
            self.jobs.start(
                project_id,
                request.action,
                handler,
                project_type=str(metadata.get("project_type", "standard_music_video")),
                reject_if_project_active=request.action in PIPELINE_ACTIONS,
                pipeline_mode=pipeline_mode,
            ),
        )


class ReferenceRerenderAction:
    action = "reference-rerender"

    def __init__(self, store: ProjectStore):
        self.store = store

    def build(self, project_id: str, request: StudioJobRequest, metadata: dict[str, Any]) -> JobHandler:
        config_path = self.store.resolve_project_path(project_id, "config.json")
        if request.reference_kind not in {"actor", "location"} or not request.reference_id:
            raise ValueError("reference-rerender requires reference_kind and reference_id")
        return build_reference_rerender_handler(
            config_path,
            reference_kind=request.reference_kind,
            reference_id=request.reference_id,
        )


class VisualConsistencyPreflightAction:
    action = "visual-consistency-preflight"

    def __init__(self, store: ProjectStore):
        self.store = store

    def build(
        self,
        project_id: str,
        request: StudioJobRequest,
        metadata: dict[str, Any],
    ) -> JobHandler:
        mode = request.visual_consistency_mode or "ingredients"
        if mode not in {"ingredients", "msr", "i2v"}:
            raise ValueError(
                "visual_consistency_mode must be ingredients, msr, or i2v",
            )
        project_dir = self.store.resolve_project_path(project_id, ".")
        plan_path = self.store.resolve_project_path(
            project_id,
            request.plan or "output/render/plans/ingredients.json",
        )
        return build_visual_consistency_preflight_handler(
            project_dir,
            plan_path=plan_path,
            mode=mode,
            preflight_mode=request.preflight_mode,
            workflow_profile=request.workflow_profile,
        )


class RecutSceneAction:
    action = "recut-scene"

    def __init__(self, store: ProjectStore):
        self.store = store

    def build(self, project_id: str, request: StudioJobRequest, metadata: dict[str, Any]) -> JobHandler:
        if (
            not request.raw_clip
            or not request.output_clip
            or request.raw_in_seconds is None
            or request.raw_out_seconds is None
        ):
            raise ValueError("recut-scene requires raw_clip, output_clip, raw_in_seconds, and raw_out_seconds")
        return build_recut_scene_handler(
            self.store.resolve_project_path(project_id, request.raw_clip),
            self.store.resolve_project_path(project_id, request.output_clip),
            raw_in_seconds=request.raw_in_seconds,
            raw_out_seconds=request.raw_out_seconds,
            exact=request.exact,
        )


class ThumbnailPrebuildAction:
    action = "thumbnail-prebuild"

    def __init__(self, store: ProjectStore):
        self.store = store

    def build(self, project_id: str, request: StudioJobRequest, metadata: dict[str, Any]) -> JobHandler:
        thumbnails = request.thumbnails or []

        def run(log: Callable[[str], None]) -> int:
            count = 0
            for thumbnail in thumbnails:
                path = str(thumbnail.get("path") or "")
                for second in thumbnail.get("times") or []:
                    thumbnail_path(self.store, project_id, path, float(second))
                    count += 1
            log(f"Generated {count} thumbnails")
            return count

        return run


class ThumbnailCleanupAction:
    action = "thumbnail-cleanup"

    def __init__(self, store: ProjectStore):
        self.store = store

    def build(self, project_id: str, request: StudioJobRequest, metadata: dict[str, Any]) -> JobHandler:
        def run(log: Callable[[str], None]) -> int:
            log("Clearing thumbnail cache")
            return self.store.clear_thumbnail_cache(project_id)

        return run


class FullAutoAction:
    action = "full-auto"

    def __init__(self, store: ProjectStore, factory: FullAutoHandlerFactory | None):
        self.store = store
        self.factory = factory

    def build(self, project_id: str, request: StudioJobRequest, metadata: dict[str, Any]) -> JobHandler:
        if metadata.get("project_type") != "full_auto":
            raise ValueError("full-auto jobs require a full_auto project")
        factory = self.factory or build_full_auto_handler
        handler = factory(
            store=self.store,
            project_id=project_id,
            payload={**dict(metadata.get("full_auto") or {}), "silent_mode": bool(metadata.get("silent_mode", False))},
        )
        return record_pipeline_state(self.store, project_id, self.action, handler)


class MovieFullAutoAction:
    action = "movie-full-auto"

    def __init__(self, store: ProjectStore):
        self.store = store

    def build(self, project_id: str, request: StudioJobRequest, metadata: dict[str, Any]) -> JobHandler:
        if metadata.get("project_type") != "movie":
            raise ValueError("movie-full-auto jobs require a movie project")
        render_plan_path = self.store.resolve_project_path(project_id, "movie/render_plan.json")
        if not render_plan_path.exists():
            raise ValueError("movie-full-auto requires movie/render_plan.json; create the movie scaffold first")
        handler = build_movie_full_auto_handler(
            store=self.store,
            project_id=project_id,
            render_plan_path=render_plan_path,
            movie_config=movie_config_from_metadata(metadata),
        )
        return record_pipeline_state(self.store, project_id, self.action, handler)


class MovieReferencesAction:
    action = "movie-references"

    def __init__(self, store: ProjectStore):
        self.store = store

    def build(self, project_id: str, request: StudioJobRequest, metadata: dict[str, Any]) -> JobHandler:
        if metadata.get("project_type") != "movie":
            raise ValueError("movie-references jobs require a movie project")
        manifest_path = self.store.resolve_project_path(project_id, "movie/references/manifest.json")
        if not manifest_path.exists():
            raise ValueError("movie-references requires movie/references/manifest.json; create the movie scaffold first")
        handler = build_movie_references_handler(
            store=self.store,
            project_id=project_id,
            movie_config=movie_config_from_metadata(metadata),
        )
        return record_pipeline_state(self.store, project_id, self.action, handler)


class MovieRenderAction:
    action = "movie-render"

    def __init__(self, store: ProjectStore):
        self.store = store

    def build(self, project_id: str, request: StudioJobRequest, metadata: dict[str, Any]) -> JobHandler:
        if metadata.get("project_type") != "movie":
            raise ValueError("movie-render jobs require a movie project")
        render_plan_msr_path = self.store.resolve_project_path(project_id, "movie/render_plan_msr.json")
        if not render_plan_msr_path.exists():
            raise ValueError("movie-render requires movie/render_plan_msr.json; run Movie full-auto or CLI MSR enrichment first")
        manifest_path = self.store.resolve_project_path(project_id, "movie/references/manifest.json")
        config = movie_config_from_metadata(metadata)
        if not movie_references_ready(manifest_path, backend=config["reference_backend"]):
            raise ValueError("movie-render requires ready movie references; run Movie references first")
        handler = build_movie_render_handler(
            store=self.store,
            project_id=project_id,
            render_plan_path=render_plan_msr_path,
            movie_config=config,
            selected_scenes=request.scenes,
        )
        return record_pipeline_state(self.store, project_id, self.action, handler)


class MovieFinalConcatAction:
    action = "movie-final-concat"

    def __init__(self, store: ProjectStore):
        self.store = store

    def build(self, project_id: str, request: StudioJobRequest, metadata: dict[str, Any]) -> JobHandler:
        if metadata.get("project_type") != "movie":
            raise ValueError("movie-final-concat jobs require a movie project")
        render_plan_msr_path = self.store.resolve_project_path(project_id, "movie/render_plan_msr.json")
        if not render_plan_msr_path.exists():
            raise ValueError("movie-final-concat requires movie/render_plan_msr.json; run Movie MSR enrichment first")
        handler = build_movie_render_handler(
            store=self.store,
            project_id=project_id,
            render_plan_path=render_plan_msr_path,
            movie_config=movie_config_from_metadata(metadata),
            concat_only=True,
        )
        return record_pipeline_state(self.store, project_id, self.action, handler)


class PipelineAction:
    action = "*"

    def __init__(self, store: ProjectStore, factory: PipelineHandlerFactory | None, provenance: ArtifactProvenancePort | None = None):
        self.store = store
        self.factory = factory
        self.provenance = provenance

    def build(self, project_id: str, request: StudioJobRequest, metadata: dict[str, Any]) -> JobHandler:
        config_path = self.store.resolve_project_path(project_id, "config.json")
        ensure_pipeline_action_available(
            self.store.project_root(project_id),
            request.action,
            request.scenes,
        )
        factory = self.factory or build_pipeline_handler
        pipeline_mode = request.pipeline_mode or pipeline_mode_from_config(config_path)
        handler = factory(config_path, request.action, scenes=request.scenes, pipeline_mode=pipeline_mode)
        return record_pipeline_state(
            self.store,
            project_id,
            request.action,
            handler,
            scenes=request.scenes,
            pipeline_mode=pipeline_mode,
            provenance=self.provenance,
        )


def record_pipeline_state(
    store: ProjectStore,
    project_id: str,
    action: str,
    handler: JobHandler,
    *,
    scenes: list[int] | None = None,
    pipeline_mode: str | None = None,
    provenance: ArtifactProvenancePort | None = None,
) -> JobHandler:
    if action not in PIPELINE_ACTIONS:
        return handler

    def run(log: Callable[[str], None]) -> Any:
        try:
            result = handler(log)
        except Exception:
            store.record_pipeline_run(project_id, action=action, stages=pipeline_state_stages(action, scenes=scenes, pipeline_mode=pipeline_mode), status="failed")
            raise
        store.record_pipeline_run(project_id, action=action, stages=pipeline_state_stages(action, scenes=scenes, pipeline_mode=pipeline_mode), status="succeeded")
        if provenance is not None:
            _record_action_fingerprint(project_id, action, scenes=scenes, provenance=provenance)
        return result

    return run


def _record_action_fingerprint(
    project_id: str,
    action: str,
    *,
    scenes: list[int] | None = None,
    provenance: ArtifactProvenancePort,
) -> None:
    """Record artifact provenance fingerprints for known pipeline actions.

    Maps pipeline actions to artifact kinds and records a fingerprint
    with the action name as a simple identifier. Full hash computation
    from output files is deferred to avoid blocking the job path.
    """
    from feverslop.domain.rebuild_policy import (
        ArtifactFingerprint,
    )

    kind = _action_to_artifact_kind(action)
    if kind is None:
        return

    scene_numbers = scenes or [None]
    for scene_number in scene_numbers:
        try:
            provenance.record_fingerprint(
                project_id=project_id,
                fingerprint=ArtifactFingerprint(
                    artifact_kind=kind,
                    scene_number=scene_number,
                    workflow_hash=_hash_action(action),
                ),
            )
        except Exception:
            logging.debug("Failed to record artifact fingerprint for %s", action, exc_info=True)


def _action_to_artifact_kind(action: str):
    from feverslop.domain.rebuild_policy import ArtifactKind

    mapping: dict[str, ArtifactKind] = {
        "rebuild-plan-timeline": ArtifactKind.RENDER_PLAN,
        "rebuild-plan": ArtifactKind.PREPARED_WORKFLOW,
        "ltx-render-scenes": ArtifactKind.SCENE_RENDER,
        "storyboard-frames": ArtifactKind.SCENE_STORYBOARD,
        "storyboard-page": ArtifactKind.SCENE_STORYBOARD,
        "storyboard": ArtifactKind.SCENE_STORYBOARD,
        "final-concat": ArtifactKind.FINAL_VIDEO,
        "concat-video-only": ArtifactKind.FINAL_VIDEO,
        "mux-original-audio": ArtifactKind.FINAL_VIDEO,
        "msr-reference-sheets": ArtifactKind.REFERENCE_SHEETS,
        "msr-references": ArtifactKind.REFERENCE_SOURCES,
        "msr-prompt-enrich": ArtifactKind.PROMPT_GENERATION,
        "msr-enrich": ArtifactKind.PROMPT_GENERATION,
        "review-ordering": ArtifactKind.REVIEW_ORDERING,
    }
    return mapping.get(action)


def _hash_action(action: str) -> str:
    return hashlib.sha256(action.encode()).hexdigest()[:16]


def pipeline_state_stages(action: str, *, scenes: list[int] | None = None, pipeline_mode: str | None = None) -> list[str]:
    try:
        options = build_pipeline_options(action, scenes=scenes, pipeline_mode=pipeline_mode)
    except ValueError:
        return [action]
    stages = options.get("stages")
    if isinstance(stages, list):
        return [str(stage) for stage in stages]
    return [action]


class StudioFullAutoConsole:
    def __init__(self, log: Callable[[str], None]):
        self.log = log

    def print(self, *values: object, **_kwargs: object) -> None:
        for line in render_log_lines(*values):
            self.log(line)

    def rule(self, title: str) -> None:
        for line in render_log_lines(str(title)):
            self.log(line)


def build_full_auto_handler(*, store: ProjectStore, project_id: str, payload: dict[str, Any], use_case_factory: Callable[[Any], Any] | None = None) -> JobHandler:
    def run(log: Callable[[str], None]) -> Any:
        from feverslop.application.full_auto import FullAutoRequest
        from feverslop.composition.full_auto import build_full_auto_use_case

        console = StudioFullAutoConsole(log)
        use_case = use_case_factory(console) if use_case_factory else build_full_auto_use_case(console=console)
        pipeline_mode = str(payload.get("pipeline_mode") or "classic")
        request = FullAutoRequest(
            idea=str(payload.get("idea") or ""),
            style=str(payload.get("song_style") or ""),
            project_name=project_id,
            projects_dir=store.projects_root,
            duration_seconds=float(payload.get("duration_seconds") or 120.0),
            width=int(payload.get("width") or 1280),
            height=int(payload.get("height") or 704),
            fps=int(payload.get("fps") or 24),
            silent_mode=bool(payload.get("silent_mode", False)),
            run_video_pipeline=True,
            runner_options={"skip_tests": True, "video_pipeline": VIDEO_PIPELINE_BY_MODE.get(pipeline_mode, "ltx_i2v")},
        )
        return run_with_stream_logging(lambda: use_case.execute(request), log).project_config_path

    return run


def pipeline_mode_from_config(config_path: Path) -> str | None:
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    value = str(config.get("video_pipeline") or "")
    if value == "ltx_msr":
        return "msr"
    if value == "ltx_ingredients":
        return "ingredients"
    if value == "ltx_i2v":
        return "classic"
    if value == "minimax-h3-r2v":
        return "minimax_h3_r2v"
    if value == "minimax-h3-t2v":
        return "minimax_h3_t2v"
    return None


def thumbnail_path(store: ProjectStore, project_id: str, path: str, at: float) -> Path:
    video_path = store.resolve_video_path(project_id, path)
    seconds = max(0.0, float(at))
    key = hashlib.sha1(f"{path}:{seconds:.2f}".encode()).hexdigest()
    thumbnail = store.thumbnail_cache_path(project_id, key)
    if thumbnail.exists():
        return thumbnail
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{seconds:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "3",
                str(thumbnail),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Could not generate thumbnail with ffmpeg") from exc
    return thumbnail
