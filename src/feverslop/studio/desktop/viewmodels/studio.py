from __future__ import annotations

import base64
import copy
import json
import mimetypes
from pathlib import Path
from typing import Any

from PySide6.QtCore import Property, QObject, QTimer, QUrl, Signal, Slot

from feverslop.studio.job_service import StudioJobRequest
from feverslop.studio.desktop.requests import project_create_request
from feverslop.studio.desktop.review_timeline import ReviewTimelineState
from feverslop.studio.projects import ArtifactRequest, RenderPlanPatch


class StudioViewModel(QObject):
    projectsChanged = Signal()
    currentProjectChanged = Signal()
    editorChanged = Signal()
    editorDirtyChanged = Signal()
    jobsChanged = Signal()
    reviewChanged = Signal()
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
        self._editor_data: Any = None
        self._editor_baseline: Any = None
        self._editor_revision: str | None = None
        self._editor_exists = False
        self._editor_dirty = False
        self._jobs: list[dict[str, Any]] = []
        self._review_path = ""
        self._review_state: ReviewTimelineState | None = None
        self._review_revision: str | None = None
        self._review_project_id = ""
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

    @Property("QVariantList", notify=currentProjectChanged)
    def artifact_entries(self) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        for category, paths in self.artifacts.items():
            for path in paths:
                suffix = str(path).lower().rsplit(".", 1)[-1] if "." in str(path) else ""
                if suffix in {"png", "jpg", "jpeg", "webp", "gif"}:
                    kind = "image"
                elif suffix in {"mp4", "mov", "webm"}:
                    kind = "video"
                elif suffix in {"mp3", "wav", "m4a", "flac", "ogg"}:
                    kind = "audio"
                elif suffix == "json":
                    kind = "json"
                else:
                    kind = "file"
                entries.append({"category": str(category), "path": str(path), "kind": kind})
        return entries

    @Property(str, notify=editorChanged)
    def editor_path(self) -> str:
        return self._editor_path

    @Property(str, notify=editorChanged)
    def editor_text(self) -> str:
        return self._editor_text

    @Property(bool, notify=editorDirtyChanged)
    def editor_dirty(self) -> bool:
        return self._editor_dirty

    @Property("QVariantList", notify=editorChanged)
    def editor_scenes(self) -> list[dict[str, Any]]:
        if not isinstance(self._editor_data, list):
            return []
        return [
            copy.deepcopy(scene)
            for scene in self._editor_data
            if isinstance(scene, dict) and "scene" in scene
        ]

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

    @Property("QVariantList", notify=reviewChanged)
    def review_items(self) -> list[dict[str, Any]]:
        if self._review_state is None:
            return []
        return self._review_state.items(list(self.artifacts.get("videos") or []))

    @Property(float, notify=reviewChanged)
    def review_duration(self) -> float:
        items = self.review_items
        return float(items[-1]["end"]) if items else 0.0

    @Property(bool, notify=reviewChanged)
    def review_dirty(self) -> bool:
        return bool(self._review_state and self._review_state.dirty)

    @Property(bool, notify=reviewChanged)
    def review_can_undo(self) -> bool:
        return bool(self._review_state and self._review_state.can_undo)

    @Property(bool, notify=reviewChanged)
    def review_can_redo(self) -> bool:
        return bool(self._review_state and self._review_state.can_redo)

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
            project = self.store.describe_project(project_id)
            changed = str(project.get("id") or "") != self.current_project_id
            self._current_project = project
            if changed:
                self._clear_editor_state()
                self._clear_review_state()
            self._set_error("")
            self.currentProjectChanged.emit()
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self._set_error(str(exc))

    @Slot("QVariantMap", result=str)
    def create_project(self, payload: dict[str, Any]) -> str:
        try:
            request = project_create_request(payload)
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

    @Slot(str, str, float, float, bool, result=bool)
    def start_recut(
        self,
        raw_clip: str,
        output_clip: str,
        raw_in_seconds: float,
        raw_out_seconds: float,
        exact: bool,
    ) -> bool:
        if not self.current_project_id:
            self._set_error("Select a project first")
            return False
        try:
            self.job_service.start_job(
                self.current_project_id,
                StudioJobRequest(
                    action="recut-scene",
                    raw_clip=raw_clip,
                    output_clip=output_clip,
                    raw_in_seconds=raw_in_seconds,
                    raw_out_seconds=raw_out_seconds,
                    exact=exact,
                ),
            )
            self.refresh_jobs()
            self._set_error("")
            return True
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self._set_error(str(exc))
            return False

    @Slot(str, str, result=bool)
    def start_reference_rerender(self, reference_kind: str, reference_id: str) -> bool:
        if not self.current_project_id:
            self._set_error("Select a project first")
            return False
        try:
            self.job_service.start_job(
                self.current_project_id,
                StudioJobRequest(
                    action="reference-rerender",
                    reference_kind=reference_kind,
                    reference_id=reference_id,
                ),
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
            data = artifact["data"]
            if data is None and not artifact.get("exists", False):
                if Path(path).name == "render_plan.json":
                    data = []
            self._editor_data = copy.deepcopy(data)
            self._editor_baseline = copy.deepcopy(data)
            self._editor_revision = artifact.get("revision")
            self._editor_exists = bool(
                artifact.get("exists", artifact["data"] is not None)
            )
            self._editor_text = json.dumps(self._editor_data, indent=2, ensure_ascii=False)
            self._set_editor_dirty(False)
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
            if Path(path).name == "render_plan.json" and not isinstance(data, list):
                raise ValueError("Render plan must be a JSON array")
            if path == self._editor_path:
                if not self._editor_exists:
                    request = ArtifactRequest(path=path, data=data, create_only=True)
                else:
                    if self._editor_revision is None:
                        current = self.store.read_artifact(self.current_project_id, path)
                        if current["data"] != self._editor_baseline:
                            raise ValueError(
                                "Artifact changed externally; reload it before saving"
                            )
                    request = ArtifactRequest(
                        path=path,
                        data=data,
                        expected_revision=self._editor_revision,
                    )
            else:
                request = ArtifactRequest(path=path, data=data, create_only=True)
            artifact = self.store.write_artifact(
                self.current_project_id,
                request,
            )
            self._editor_path = path
            self._editor_data = copy.deepcopy(artifact["data"])
            self._editor_baseline = copy.deepcopy(artifact["data"])
            self._editor_revision = artifact.get("revision")
            self._editor_exists = True
            self._editor_text = json.dumps(self._editor_data, indent=2, ensure_ascii=False)
            self._set_editor_dirty(False)
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

    @Slot(str)
    def set_json_editor_draft(self, text: str) -> None:
        self._editor_text = text
        clean_text = json.dumps(self._editor_data, indent=2, ensure_ascii=False)
        self._set_editor_dirty(text != clean_text)

    @Slot(result=bool)
    @Slot(str, result=bool)
    def refresh_render_plan_editor(self, path: str = "") -> bool:
        render_plans = self.artifacts.get("render_plans") or []
        if not render_plans:
            return False
        target_path = path or str(render_plans[0])
        if target_path not in render_plans:
            return False
        raw_matches = self._editor_path == target_path
        review_matches = self._review_state is not None and self._review_path == target_path
        if not raw_matches and not review_matches:
            return False
        blocked: list[str] = []
        if raw_matches and self._editor_dirty:
            blocked.append("Disk changed; save/reload raw draft")
        if review_matches and self._review_state is not None and self._review_state.dirty:
            blocked.append("Disk changed; save/reload review")
        refresh_raw = raw_matches and not self._editor_dirty
        refresh_review = review_matches and self._review_state is not None and not self._review_state.dirty
        artifact = None
        try:
            if refresh_raw or refresh_review:
                artifact = self.store.read_artifact(self.current_project_id, target_path)
            if refresh_raw and artifact is not None:
                self._editor_data = copy.deepcopy(artifact["data"])
                self._editor_baseline = copy.deepcopy(artifact["data"])
                self._editor_revision = artifact.get("revision")
                self._editor_exists = bool(
                    artifact.get("exists", artifact["data"] is not None)
                )
                self._editor_text = json.dumps(
                    self._editor_data,
                    indent=2,
                    ensure_ascii=False,
                )
                self._set_editor_dirty(False)
                self.editorChanged.emit()
            if refresh_review and artifact is not None:
                self._review_state = ReviewTimelineState.from_document(artifact["data"])
                self._review_revision = artifact.get("revision")
                self.reviewChanged.emit()
            self._set_error("; ".join(blocked))
            return not blocked and (refresh_raw or refresh_review)
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self._set_error(str(exc))
            return False

    @Slot(str, result=str)
    def preferred_artifact(self, category: str) -> str:
        paths = self.artifacts.get(category) or []
        return str(paths[0]) if paths else ""

    @Slot(result=bool)
    def load_review_timeline(self) -> bool:
        if not self.current_project_id:
            self._set_error("Select a project first")
            return False
        path = self.preferred_artifact("render_plans")
        if not path:
            self._set_error("No render plan found")
            return False
        if (
            self._review_state is not None
            and self._review_state.dirty
            and self._review_project_id == self.current_project_id
            and self._review_path == path
        ):
            self._set_error("Dirty review exists; save/reload review before loading")
            return False
        try:
            artifact = self.store.read_artifact(self.current_project_id, path)
            self._review_path = path
            self._review_state = ReviewTimelineState.from_document(artifact["data"])
            self._review_revision = artifact.get("revision")
            self._review_project_id = self.current_project_id
            self._set_error("")
            self.reviewChanged.emit()
            return True
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self._set_error(str(exc))
            return False

    @Slot(int, int, result=bool)
    def move_review_scene(self, source_index: int, target_index: int) -> bool:
        changed = bool(self._review_state and self._review_state.move(source_index, target_index))
        if changed:
            self.reviewChanged.emit()
        return changed

    @Slot(int, float, float, result=bool)
    def trim_review_scene(self, scene: int, raw_in_seconds: float, raw_out_seconds: float) -> bool:
        changed = bool(self._review_state and self._review_state.trim(scene, raw_in_seconds, raw_out_seconds))
        if changed:
            self.reviewChanged.emit()
        return changed

    @Slot(result=bool)
    def undo_review_timeline(self) -> bool:
        changed = bool(self._review_state and self._review_state.undo())
        if changed:
            self.reviewChanged.emit()
        return changed

    @Slot(result=bool)
    def redo_review_timeline(self) -> bool:
        changed = bool(self._review_state and self._review_state.redo())
        if changed:
            self.reviewChanged.emit()
        return changed

    @Slot(result=bool)
    def save_review_timeline(self) -> bool:
        if self._review_state is None or not self._review_path:
            self._set_error("Load a review timeline first")
            return False
        try:
            artifact = self.store.write_artifact(
                self.current_project_id,
                ArtifactRequest(
                    path=self._review_path,
                    data=self._review_state.document(),
                    expected_revision=self._review_revision,
                ),
            )
            self._review_revision = artifact.get("revision")
            self._review_state.mark_saved()
            self._set_error("")
            self.reviewChanged.emit()
            return True
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self._set_error(str(exc))
            return False

    @Slot(str, int, "QVariantMap", result=bool)
    def patch_render_scene(self, path: str, scene: int, updates: dict[str, Any]) -> bool:
        if not self.current_project_id:
            self._set_error("Select a project first")
            return False
        if path != self._editor_path:
            self._set_error("Load target before patching")
            return False
        if self._editor_dirty:
            self._set_error("Save or reload raw draft first")
            return False
        try:
            self.store.patch_render_plan(
                self.current_project_id,
                RenderPlanPatch(
                    path=path,
                    scene=scene,
                    updates=dict(updates),
                    expected_revision=self._editor_revision,
                ),
            )
            self.refresh_render_plan_editor(path)
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

    @Slot(str, str, result=bool)
    def import_image(self, source_url: str, target_path: str) -> bool:
        if not self.current_project_id:
            self._set_error("Select a project first")
            return False
        try:
            source_path = QUrl(source_url).toLocalFile()
            mime_type = mimetypes.guess_type(source_path)[0] or "image/png"
            if not mime_type.startswith("image/"):
                raise ValueError("Select a PNG, JPEG, or WebP image")
            encoded = base64.b64encode(Path(source_path).read_bytes()).decode("ascii")
            self.store.write_media_data_url(
                self.current_project_id,
                target_path,
                f"data:{mime_type};base64,{encoded}",
            )
            self.select_project(self.current_project_id)
            self._set_error("")
            return True
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self._set_error(str(exc))
            return False

    @Slot(str, result=bool)
    def import_audio(self, source_url: str) -> bool:
        if not self.current_project_id:
            self._set_error("Select a project first")
            return False
        try:
            source_path = Path(QUrl(source_url).toLocalFile())
            mime_type = mimetypes.guess_type(source_path.name)[0] or "application/octet-stream"
            with source_path.open("rb") as source:
                self.store.store_audio_upload(
                    self.current_project_id,
                    source_path.name,
                    mime_type,
                    source,
                )
            self.select_project(self.current_project_id)
            self._set_error("")
            return True
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self._set_error(str(exc))
            return False

    def _set_error(self, message: str) -> None:
        if message == self._error:
            return
        self._error = message
        self.errorChanged.emit()

    def _set_editor_dirty(self, dirty: bool) -> None:
        if dirty == self._editor_dirty:
            return
        self._editor_dirty = dirty
        self.editorDirtyChanged.emit()

    def _clear_editor_state(self) -> None:
        self._editor_path = ""
        self._editor_text = ""
        self._editor_data = None
        self._editor_baseline = None
        self._editor_revision = None
        self._editor_exists = False
        self._set_editor_dirty(False)
        self.editorChanged.emit()

    def _clear_review_state(self) -> None:
        self._review_path = ""
        self._review_state = None
        self._review_revision = None
        self._review_project_id = ""
        self.reviewChanged.emit()
