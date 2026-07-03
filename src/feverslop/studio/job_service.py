from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Protocol

from feverslop.studio.jobs import (
    PIPELINE_ACTIONS,
    JobHandler,
    JobRegistry,
    build_pipeline_handler,
    build_pipeline_options,
    build_recut_scene_handler,
    build_reference_rerender_handler,
    run_with_stream_logging,
)
from feverslop.studio.logging import render_log_lines
from feverslop.studio.projects import ProjectStore


FullAutoHandlerFactory = Callable[..., Any]
PipelineHandlerFactory = Callable[..., Any]


@dataclass(frozen=True)
class StudioJobRequest:
    action: str
    scenes: list[int] | None = None
    pipeline_mode: str | None = None
    thumbnails: list[dict[str, Any]] | None = None
    reference_kind: str | None = None
    reference_id: str | None = None
    raw_clip: str | None = None
    output_clip: str | None = None
    raw_in_seconds: float | None = None
    raw_out_seconds: float | None = None
    exact: bool = False


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
        full_auto_handler: FullAutoHandlerFactory | None = None,
        pipeline_handler: PipelineHandlerFactory | None = None,
    ):
        self.store = store
        self.jobs = jobs
        self.registry = ActionRegistry(
            [
                ReferenceRerenderAction(store),
                RecutSceneAction(store),
                ThumbnailPrebuildAction(store),
                ThumbnailCleanupAction(store),
                FullAutoAction(store, full_auto_handler),
                MovieReferencesAction(store),
                MovieFullAutoAction(store),
            ],
            PipelineAction(store, pipeline_handler),
        )

    def start_job(self, project_id: str, request: StudioJobRequest) -> dict[str, Any]:
        metadata = self.store.project_metadata(project_id)
        handler = self.registry.resolve(request.action).build(project_id, request, metadata)
        return self.jobs.get(
            self.jobs.start(
                project_id,
                request.action,
                handler,
                project_type=str(metadata.get("project_type", "standard_music_video")),
                reject_if_project_active=request.action in PIPELINE_ACTIONS,
            )
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
        return factory(store=self.store, project_id=project_id, payload=metadata.get("full_auto") or {})


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
        handler = build_movie_full_auto_handler(store=self.store, project_id=project_id, render_plan_path=render_plan_path)
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
        handler = build_movie_references_handler(store=self.store, project_id=project_id)
        return record_pipeline_state(self.store, project_id, self.action, handler)


class PipelineAction:
    action = "*"

    def __init__(self, store: ProjectStore, factory: PipelineHandlerFactory | None):
        self.store = store
        self.factory = factory

    def build(self, project_id: str, request: StudioJobRequest, metadata: dict[str, Any]) -> JobHandler:
        config_path = self.store.resolve_project_path(project_id, "config.json")
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
        )


def record_pipeline_state(
    store: ProjectStore,
    project_id: str,
    action: str,
    handler: JobHandler,
    *,
    scenes: list[int] | None = None,
    pipeline_mode: str | None = None,
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
        return result

    return run


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
            run_video_pipeline=True,
            runner_options={"skip_tests": True, "video_pipeline": "ltx_msr" if pipeline_mode == "msr" else "ltx_i2v"},
        )
        return run_with_stream_logging(lambda: use_case.execute(request), log).project_config_path

    return run


def build_movie_full_auto_handler(*, store: ProjectStore, project_id: str, render_plan_path: Path) -> JobHandler:
    def run(log: Callable[[str], None]) -> Any:
        project_dir = store.resolve_project_path(project_id, ".").resolve()
        log("[MoviePipeline] Stage: Story-Arch Complete")
        log("[MoviePipeline] Stage: Render Plan Ready")
        log("[Krea2_Adapter] Preparing visual consistency references")
        manifest_path = ensure_movie_references(project_dir)
        log(f"[Krea2_Adapter] Reference sheets ready: {manifest_path}")
        patched_workflow = patch_movie_msr_workflow(project_dir)
        log(f"[WorkflowPatcher] Movie MSR workflow patched for LTX native audio: {patched_workflow}")
        log("[LTX_MSR_Movie_Adapter] Rendering with LTX 2.3 native audio; no custom audio track supplied")
        final_video = build_movie_visual_adapter(project_dir, patched_workflow).render_movie(
            project_dir=project_dir,
            render_plan_path=render_plan_path,
        )
        log(f"[MoviePipeline] Stage: Movie Complete: {final_video}")
        return final_video

    return run


def ensure_movie_references(project_dir: Path) -> Path:
    manifest_path = project_dir / "movie" / "references" / "manifest.json"
    if movie_references_ready(manifest_path):
        return manifest_path
    return build_movie_reference_generator().generate(project_dir=project_dir)


def movie_references_ready(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actors = manifest.get("actors") or []
    locations = manifest.get("locations") or []
    if not actors or not locations:
        return False
    return all(str(item.get("msr_sheet_path") or "").strip() for item in [*actors, *locations])


def build_movie_references_handler(*, store: ProjectStore, project_id: str) -> JobHandler:
    def run(log: Callable[[str], None]) -> Any:
        project_dir = store.resolve_project_path(project_id, ".").resolve()
        log("[MoviePipeline] Stage: Movie references")
        manifest_path = build_movie_reference_generator().generate(project_dir=project_dir)
        log(f"[Krea2_Adapter] Reference sheets ready: {manifest_path}")
        return manifest_path

    return run


def build_movie_reference_generator():
    backend = str(os.environ.get("FEVERSLOP_MOVIE_REFERENCE_BACKEND") or "local").strip().lower()
    if backend in {"", "local", "placeholder"}:
        from feverslop.adapters.movie_references import LocalMovieImageBackend
        from feverslop.application.movie_references import MovieReferenceSheetGenerator

        local = LocalMovieImageBackend()
        return MovieReferenceSheetGenerator(backend=local, edit_backend=local)
    if backend != "comfyui":
        raise ValueError("FEVERSLOP_MOVIE_REFERENCE_BACKEND must be local or comfyui")

    from feverslop.adapters.comfyui_client import ComfyUIClient
    from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver
    from feverslop.adapters.comfyui_rendering import ComfyUIImageBackend
    from feverslop.application.movie_references import MovieReferenceSheetGenerator
    from feverslop.config.app_config import AppConfig
    from feverslop.ports.rendering import WorkflowAnchorConfig

    app_config = AppConfig.load(os.environ.get("FEVERSLOP_APP_CONFIG", "app_config.json"))
    client = ComfyUIClient(
        base_url=app_config.comfyui.base_url,
        prompt_timeout_seconds=app_config.comfyui.prompt_timeout_seconds,
    )
    resolver = ComfyUIModelResolver(client, overrides=app_config.comfyui.model_overrides)
    hero = ComfyUIImageBackend(
        client=client,
        workflow_path=os.environ.get("FEVERSLOP_MOVIE_HERO_WORKFLOW", str(Path("workflows") / "image_t2i_startframe_krea_v1.json")),
        output_dir=Path("."),
        model_resolver=resolver,
    )
    edit = ComfyUIImageBackend(
        client=client,
        workflow_path=os.environ.get("FEVERSLOP_MOVIE_EDIT_WORKFLOW", str(Path("workflows") / "image_edit_flux2_klein_1ref_v1.json")),
        output_dir=Path("."),
        model_resolver=resolver,
    )
    return MovieReferenceSheetGenerator(
        backend=hero,
        edit_backend=edit,
        hero_anchors=WorkflowAnchorConfig(positive_prompt_title="#PROMPT_POSITIVE"),
        edit_anchors=WorkflowAnchorConfig(positive_prompt_title="#PROMPT_POSITIVE", reference_image_title="#IMAGE_1"),
    )


def build_movie_visual_adapter(project_dir: Path, workflow_path: Path):
    backend = str(os.environ.get("FEVERSLOP_MOVIE_RENDER_BACKEND") or "local").strip().lower()
    if backend in {"", "local", "placeholder"}:
        from feverslop.adapters.movie_visual import LocalMovieVisualAdapter

        return LocalMovieVisualAdapter()
    if backend != "comfyui":
        raise ValueError("FEVERSLOP_MOVIE_RENDER_BACKEND must be local or comfyui")

    from feverslop.adapters.comfyui_client import ComfyUIClient
    from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver
    from feverslop.adapters.movie_visual import ComfyUIMovieVisualAdapter
    from feverslop.config.app_config import AppConfig

    app_config = AppConfig.load(os.environ.get("FEVERSLOP_APP_CONFIG", "app_config.json"))
    client = ComfyUIClient(
        base_url=app_config.comfyui.base_url,
        prompt_timeout_seconds=app_config.comfyui.prompt_timeout_seconds,
    )
    return ComfyUIMovieVisualAdapter(
        client=client,
        workflow_path=workflow_path,
        model_resolver=ComfyUIModelResolver(client, overrides=app_config.comfyui.model_overrides),
    )


def patch_movie_msr_workflow(project_dir: Path, *, template_path: Path = Path("workflows") / "video_ltxv_msr_1actor_1background_v1.json") -> Path:
    from feverslop.adapters.movie_workflow import MovieWorkflowPatcher

    if not template_path.exists():
        raise FileNotFoundError(f"Movie MSR workflow template not found: {template_path}")
    workflow = json.loads(template_path.read_text(encoding="utf-8"))
    patched = MovieWorkflowPatcher().strip_audio_inputs(workflow)
    output_path = project_dir / "movie" / "workflows" / "video_ltxv_msr_movie_native_audio.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(patched, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def pipeline_mode_from_config(config_path: Path) -> str | None:
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    value = str(config.get("video_pipeline") or "")
    if value == "ltx_msr":
        return "msr"
    if value == "ltx_i2v":
        return "classic"
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
