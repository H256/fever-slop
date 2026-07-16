from __future__ import annotations

import io
import json
from dataclasses import fields
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from feverslop.studio.job_service import StudioJobRequest, StudioJobService
from feverslop.studio.jobs import JobRegistry
from feverslop.studio.desktop.requests import project_create_request
from feverslop.studio.projects import ProjectStore


class HarnessResponse:
    def __init__(self, status_code: int, data: Any):
        self.status_code = status_code
        self._data = data

    @property
    def text(self) -> str:
        return json.dumps(self._data, default=str)

    def json(self) -> Any:
        return self._data


class NativeStudioHarness:
    """Exercises the native Studio service boundary without an HTTP adapter."""

    def __init__(self, projects_root: str | Path, *, full_auto_handler=None, pipeline_handler=None):
        self.store = ProjectStore(projects_root)
        self.jobs = JobRegistry()
        self.service = StudioJobService(
            store=self.store,
            jobs=self.jobs,
            full_auto_handler=full_auto_handler,
            pipeline_handler=pipeline_handler,
        )

    def post(self, route: str, *, json: dict[str, Any] | None = None, files=None) -> HarnessResponse:
        try:
            if route == "/api/projects":
                payload = dict(json or {})
                return HarnessResponse(200, self.store.create_project(project_create_request(payload)))
            if route.endswith("/upload-audio"):
                project_id = route.split("/")[3]
                filename, content, content_type = files["file"]
                return HarnessResponse(
                    200,
                    self.store.store_audio_upload(project_id, filename, content_type, io.BytesIO(content)),
                )
            if route.endswith("/jobs"):
                project_id = route.split("/")[3]
                payload = dict(json or {})
                allowed = {field.name for field in fields(StudioJobRequest)}
                request = StudioJobRequest(**{key: value for key, value in payload.items() if key in allowed})
                return HarnessResponse(200, self.service.start_job(project_id, request))
            raise ValueError(f"Unsupported native test route: {route}")
        except Exception as exc:  # noqa: BLE001 - test boundary mirrors UI error handling
            return HarnessResponse(400, {"detail": str(exc)})

    def get(self, route: str) -> HarnessResponse:
        try:
            parsed = urlsplit(route)
            if parsed.path.startswith("/api/jobs/"):
                return HarnessResponse(200, self.jobs.get(parsed.path.rsplit("/", 1)[-1]))
            if parsed.path == "/api/processes":
                project_id = parse_qs(parsed.query).get("project_id", [None])[0]
                return HarnessResponse(200, self.jobs.list(project_id))
            raise ValueError(f"Unsupported native test route: {route}")
        except Exception as exc:  # noqa: BLE001 - test boundary mirrors UI error handling
            return HarnessResponse(400, {"detail": str(exc)})
