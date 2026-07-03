from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from feverslop.studio.jobs import JobRegistry
from feverslop.studio.job_service import (
    StudioFullAutoConsole as _StudioFullAutoConsole,  # noqa: F401
    StudioJobRequest,
    StudioJobService,
    build_full_auto_handler,  # noqa: F401
    thumbnail_path,
)
from feverslop.studio.projects import ArtifactRequest, ProjectCreateRequest, ProjectStore, RenderPlanPatch, StudioPathError


class ArtifactPayload(BaseModel):
    path: str
    data: Any


class MediaPayload(BaseModel):
    path: str
    data_url: str


class RenderPlanPatchPayload(BaseModel):
    path: str
    scene: int
    updates: dict[str, Any]


class ProjectCreatePayload(BaseModel):
    project_type: str
    name: str
    silent_mode: Any = False
    source_type: str = "short_story"
    story_text: str = ""
    desired_length: float = 60.0
    movie_mode: str = "scaffold"
    idea: str = ""
    song_style: str = ""
    duration_seconds: float = 120.0
    width: int = 1280
    height: int = 704
    fps: int = 24
    pipeline_mode: str = "classic"


class JobPayload(BaseModel):
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


FullAutoHandlerFactory = Callable[..., Any]
PipelineHandlerFactory = Callable[..., Any]


def create_app(
    projects_root: str | Path = "projects",
    *,
    full_auto_handler: FullAutoHandlerFactory | None = None,
    pipeline_handler: PipelineHandlerFactory | None = None,
) -> FastAPI:
    app = FastAPI(title="FeverSlop Studio")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    store = ProjectStore(projects_root)
    jobs = JobRegistry()
    job_service = StudioJobService(
        store=store,
        jobs=jobs,
        full_auto_handler=full_auto_handler,
        pipeline_handler=pipeline_handler,
    )

    @app.get("/api/projects")
    def list_projects():
        return store.list_projects()

    @app.post("/api/projects")
    def create_project(payload: ProjectCreatePayload):
        return _safe(
            lambda: store.create_project(
                ProjectCreateRequest(
                    project_type=payload.project_type,
                    name=payload.name,
                    silent_mode=_validated_silent_mode(payload.silent_mode),
                    source_type=payload.source_type,
                    story_text=payload.story_text,
                    desired_length=payload.desired_length,
                    movie_mode=payload.movie_mode,
                    idea=payload.idea,
                    song_style=payload.song_style,
                    duration_seconds=payload.duration_seconds,
                    width=payload.width,
                    height=payload.height,
                    fps=payload.fps,
                    pipeline_mode=payload.pipeline_mode,
                )
            )
        )

    @app.get("/api/projects/{project_id}")
    def get_project(project_id: str):
        try:
            return store.describe_project(project_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/projects/{project_id}/artifacts")
    def list_artifacts(project_id: str):
        return _safe(lambda: store.list_artifacts(project_id))

    @app.get("/api/projects/{project_id}/artifact")
    def read_artifact(project_id: str, path: str):
        return _safe(lambda: store.read_artifact(project_id, path))

    @app.put("/api/projects/{project_id}/artifact")
    def write_artifact(project_id: str, payload: ArtifactPayload):
        return _safe(lambda: store.write_artifact(project_id, ArtifactRequest(path=payload.path, data=payload.data)))

    @app.patch("/api/projects/{project_id}/render-plan")
    def patch_render_plan(project_id: str, payload: RenderPlanPatchPayload):
        return _safe(
            lambda: store.patch_render_plan(
                project_id,
                RenderPlanPatch(path=payload.path, scene=payload.scene, updates=payload.updates),
            )
        )

    @app.get("/api/projects/{project_id}/media")
    def get_media(project_id: str, path: str):
        return _safe(lambda: FileResponse(store.resolve_media_path(project_id, path)))

    @app.put("/api/projects/{project_id}/media")
    def write_media(project_id: str, payload: MediaPayload):
        return _safe(lambda: store.write_media_data_url(project_id, payload.path, payload.data_url))

    @app.post("/api/projects/{project_id}/upload-audio")
    def upload_audio(project_id: str, file: UploadFile = File(...)):
        return _safe(lambda: store.store_audio_upload(project_id, file.filename or "", file.content_type or "", file.file))

    @app.get("/api/projects/{project_id}/thumbnail")
    def get_thumbnail(project_id: str, path: str, at: float = 0.0):
        def create():
            return FileResponse(_thumbnail_path(store, project_id, path, at), media_type="image/jpeg")

        return _safe(create)

    @app.post("/api/projects/{project_id}/jobs")
    def start_job(project_id: str, payload: JobPayload):
        return _safe(
            lambda: job_service.start_job(
                project_id,
                StudioJobRequest(
                    action=payload.action,
                    scenes=payload.scenes,
                    pipeline_mode=payload.pipeline_mode,
                    thumbnails=payload.thumbnails,
                    reference_kind=payload.reference_kind,
                    reference_id=payload.reference_id,
                    raw_clip=payload.raw_clip,
                    output_clip=payload.output_clip,
                    raw_in_seconds=payload.raw_in_seconds,
                    raw_out_seconds=payload.raw_out_seconds,
                    exact=payload.exact,
                ),
            )
        )

    @app.get("/api/jobs")
    def list_jobs(project_id: str | None = None):
        return jobs.list(project_id)

    @app.get("/api/processes")
    def list_processes(project_id: str | None = None):
        return jobs.list(project_id)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        try:
            return jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}") from exc

    @app.get("/api/jobs/{job_id}/logs")
    def stream_job_logs(job_id: str):
        def events():
            offset = 0
            while True:
                try:
                    job = jobs.get(job_id)
                except KeyError:
                    yield f"event: error\ndata: {json.dumps({'error': 'job not found'})}\n\n"
                    return
                logs = job.get("logs") or []
                for line in logs[offset:]:
                    yield f"data: {json.dumps({'line': line})}\n\n"
                offset = len(logs)
                if job.get("status") in {"succeeded", "failed"}:
                    yield f"event: status\ndata: {json.dumps({'status': job.get('status')})}\n\n"
                    return
                time.sleep(1)

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def _safe(fn):
    try:
        return fn()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (StudioPathError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _validated_silent_mode(value: Any) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError("silent_mode must be a boolean")
    return value


def _thumbnail_path(store: ProjectStore, project_id: str, path: str, at: float) -> Path:
    return thumbnail_path(store, project_id, path, at)


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("feverslop.studio.server:app", host="127.0.0.1", port=8765, reload=True)


if __name__ == "__main__":
    main()
