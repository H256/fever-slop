from __future__ import annotations

import json
from dataclasses import fields
from typing import Any

from PySide6.QtCore import Property, QObject, QTimer, QUrl, Signal, Slot

from feverslop.studio.job_service import StudioJobRequest
from feverslop.studio.projects import ArtifactRequest, ProjectCreateRequest, RenderPlanPatch


class StudioViewModel(QObject):
    projectsChanged = Signal()
    currentProjectChanged = Signal()
    editorChanged = Signal()
    jobsChanged = Signal()
    errorChanged = Signal()

    def __init__(self, *, store: Any, jobs: Any, job_service: Any, parent: QObject | None = None):
        super().__init__(parent)
        self.store = store
        self._job_registry = jobs
        self.job_service = job_service
        self._projects: list[dict[str, Any]] = []
        self._current_project: dict[str, Any] = {}
        self._editor_path = ""
        self._editor_text = ""
        self._jobs: list[dict[str, Any]] = []
        self._error = ""
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(500)
        self._poll_timer.timeout.connect(self.refresh_jobs)

    @Property("QVariantList", notify=projectsChanged)
    def projects(self) -> list[dict[str, Any]]:
        return self._projects

    @Property("QVariantMap", notify=currentProjectChanged)
    def current_project(self) -> dict[str, Any]:
        return self._current_project

    @Property(str, notify=currentProjectChanged)
    def current_project_id(self) -> str:
        return str(self._current_project.get("id") or "")

    @Property("QVariantMap", notify=currentProjectChanged)
    def artifacts(self) -> dict[str, list[str]]:
        return dict(self._current_project.get("artifacts") or {})

    @Property(str, notify=editorChanged)
    def editor_path(self) -> str:
        return self._editor_path

    @Property(str, notify=editorChanged)
    def editor_text(self) -> str:
        return self._editor_text

    @Property("QVariantList", notify=jobsChanged)
    def jobs(self) -> list[dict[str, Any]]:
        return self._jobs

    @Property(str, notify=jobsChanged)
    def job_logs(self) -> str:
        if not self._jobs:
            return "Ready."
        logs = self._jobs[0].get("logs") or self._jobs[0].get("recent_logs") or []
        return "\n".join(str(line) for line in logs)

    @Property("QVariantMap", notify=jobsChanged)
    def active_job(self) -> dict[str, Any]:
        return next((job for job in self._jobs if job.get("status") in {"queued", "running"}), {})

    @Property(str, notify=errorChanged)
    def error(self) -> str:
        return self._error

    @Slot()
    def refresh_projects(self) -> None:
        try:
            self._projects = self.store.list_projects()
            self._set_error("")
            self.projectsChanged.emit()
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self._set_error(str(exc))

    @Slot(str)
    def select_project(self, project_id: str) -> None:
        try:
            self._current_project = self.store.describe_project(project_id)
            self._set_error("")
            self.currentProjectChanged.emit()
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self._set_error(str(exc))

    @Slot("QVariantMap", result=str)
    def create_project(self, payload: dict[str, Any]) -> str:
        try:
            allowed = {field.name for field in fields(ProjectCreateRequest)}
            request = ProjectCreateRequest(**{key: value for key, value in dict(payload).items() if key in allowed})
            project = self.store.create_project(request)
            self.refresh_projects()
            self.select_project(str(project["id"]))
            return str(project["id"])
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self._set_error(str(exc))
            return ""

    @Slot()
    def start_polling(self) -> None:
        self._poll_timer.start()

    @Slot()
    def refresh_jobs(self) -> None:
        if not self.current_project_id or not hasattr(self.jobs_service, "list"):
            return
        try:
            self._jobs = self.jobs_service.list(self.current_project_id)
            self.jobsChanged.emit()
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self._set_error(str(exc))

    @Slot(str, "QVariantList", result=bool)
    def start_job(self, action: str, scenes: list[Any] | None = None) -> bool:
        if not self.current_project_id:
            self._set_error("Select a project first")
            return False
        try:
            selected_scenes = [int(scene) for scene in scenes or []]
            self.job_service.start_job(
                self.current_project_id,
                StudioJobRequest(action=action, scenes=selected_scenes or None),
            )
            self.refresh_jobs()
            self._set_error("")
            return True
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self._set_error(str(exc))
            return False

    @property
    def jobs_service(self) -> Any:
        return self._job_registry

    @Slot(str)
    def load_json_artifact(self, path: str) -> None:
        if not self.current_project_id:
            self._set_error("Select a project first")
            return
        try:
            artifact = self.store.read_artifact(self.current_project_id, path)
            self._editor_path = path
            self._editor_text = json.dumps(artifact["data"], indent=2, ensure_ascii=False)
            self._set_error("")
            self.editorChanged.emit()
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self._set_error(str(exc))

    @Slot(str, str, result=bool)
    def save_json_artifact(self, path: str, text: str) -> bool:
        if not self.current_project_id:
            self._set_error("Select a project first")
            return False
        try:
            data = json.loads(text)
            artifact = self.store.write_artifact(
                self.current_project_id,
                ArtifactRequest(path=path, data=data),
            )
            self._editor_path = path
            self._editor_text = json.dumps(artifact["data"], indent=2, ensure_ascii=False)
            self._current_project = self.store.describe_project(self.current_project_id)
            self._set_error("")
            self.editorChanged.emit()
            self.currentProjectChanged.emit()
            return True
        except json.JSONDecodeError as exc:
            self._set_error(f"Invalid JSON: {exc.msg} at line {exc.lineno}, column {exc.colno}")
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self._set_error(str(exc))
        return False

    @Slot(str, int, "QVariantMap", result=bool)
    def patch_render_scene(self, path: str, scene: int, updates: dict[str, Any]) -> bool:
        if not self.current_project_id:
            self._set_error("Select a project first")
            return False
        try:
            self.store.patch_render_plan(
                self.current_project_id,
                RenderPlanPatch(path=path, scene=scene, updates=dict(updates)),
            )
            self.load_json_artifact(path)
            return True
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self._set_error(str(exc))
            return False

    @Slot(str, result=str)
    def media_url(self, path: str) -> str:
        if not self.current_project_id or not path:
            return ""
        try:
            media_path = self.store.resolve_media_path(self.current_project_id, path)
            self._set_error("")
            return QUrl.fromLocalFile(str(media_path)).toString()
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self._set_error(str(exc))
            return ""

    def _set_error(self, message: str) -> None:
        if message == self._error:
            return
        self._error = message
        self.errorChanged.emit()
