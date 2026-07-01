from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from feverslop.studio.jobs import JobRegistry, build_pipeline_handler
from feverslop.studio.projects import ArtifactRequest, ProjectStore, RenderPlanPatch, StudioPathError


class ArtifactPayload(BaseModel):
    path: str
    data: Any


class RenderPlanPatchPayload(BaseModel):
    path: str
    scene: int
    updates: dict[str, Any]


class JobPayload(BaseModel):
    action: str
    scenes: list[int] | None = None


def create_app(projects_root: str | Path = "projects") -> FastAPI:
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

    @app.get("/api/projects")
    def list_projects():
        return store.list_projects()

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

    @app.post("/api/projects/{project_id}/jobs")
    def start_job(project_id: str, payload: JobPayload):
        def create():
            config_path = store.resolve_project_path(project_id, "config.json")
            handler = build_pipeline_handler(config_path, payload.action, scenes=payload.scenes)
            return jobs.get(jobs.start(project_id, payload.action, handler))

        return _safe(create)

    @app.get("/api/jobs")
    def list_jobs(project_id: str | None = None):
        return jobs.list(project_id)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        try:
            return jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}") from exc

    return app


def _safe(fn):
    try:
        return fn()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (StudioPathError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("feverslop.studio.server:app", host="127.0.0.1", port=8765, reload=True)


if __name__ == "__main__":
    main()
