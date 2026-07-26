from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from PySide6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QObject,
    Property,
    QUrl,
    Qt,
    Signal,
    Slot,
)

from feverslop.domain.scene_workspace import SceneWorkspaceItem
from feverslop.ports.scene_documents import SceneDocumentConflict, SceneLtxPromptField


class SceneListModel(QAbstractListModel):
    SceneNumberRole = Qt.ItemDataRole.UserRole + 1
    StartSecondsRole = SceneNumberRole + 1
    EndSecondsRole = SceneNumberRole + 2
    PerformanceStateRole = SceneNumberRole + 3
    StatusRole = SceneNumberRole + 4
    ThumbnailUrlRole = SceneNumberRole + 5
    ShotDescriptionRole = SceneNumberRole + 6
    ImagePromptRole = SceneNumberRole + 7
    VideoPromptRole = SceneNumberRole + 8
    VideoPromptFieldRole = SceneNumberRole + 9
    ReferenceIdsRole = SceneNumberRole + 10
    SelectedRole = SceneNumberRole + 11

    _ROLE_NAMES = {
        SceneNumberRole: b"sceneNumber",
        StartSecondsRole: b"startSeconds",
        EndSecondsRole: b"endSeconds",
        PerformanceStateRole: b"performanceState",
        StatusRole: b"status",
        ThumbnailUrlRole: b"thumbnailUrl",
        ShotDescriptionRole: b"shotDescription",
        ImagePromptRole: b"imagePrompt",
        VideoPromptRole: b"videoPrompt",
        VideoPromptFieldRole: b"videoPromptField",
        ReferenceIdsRole: b"referenceIds",
        SelectedRole: b"selected",
    }

    def __init__(
        self,
        *,
        thumbnail_url: Callable[[str], str] | None = None,
        video_thumbnail_url: Callable[[str], str] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._items: list[SceneWorkspaceItem] = []
        self._selected: set[int] = set()
        self._video_prompt_fields: dict[int, str] = {}
        self._thumbnail_url = thumbnail_url or _local_file_url
        self._video_thumbnail_url = video_thumbnail_url or (lambda _path: "")

    def roleNames(self) -> dict[int, bytes]:  # noqa: N802 - Qt API
        return dict(self._ROLE_NAMES)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802 - Qt API
        return 0 if parent.isValid() else len(self._items)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or not 0 <= index.row() < len(self._items):
            return None
        item = self._items[index.row()]
        video_prompt_field, video_prompt = self._selected_video_prompt(item)
        values = {
            self.SceneNumberRole: item.scene_number,
            self.StartSecondsRole: item.start_seconds,
            self.EndSecondsRole: item.end_seconds,
            self.PerformanceStateRole: item.performance_state,
            self.StatusRole: item.status,
            self.ThumbnailUrlRole: self._thumbnail(item),
            self.ShotDescriptionRole: item.shot_description,
            self.ImagePromptRole: item.image_prompt,
            self.VideoPromptRole: video_prompt,
            self.VideoPromptFieldRole: video_prompt_field,
            self.ReferenceIdsRole: list(item.reference_ids),
            self.SelectedRole: item.scene_number in self._selected,
        }
        return values.get(role)

    def replace(self, items: Iterable[SceneWorkspaceItem]) -> None:
        self.beginResetModel()
        self._items = list(items)
        valid_numbers = {item.scene_number for item in self._items}
        self._selected.intersection_update(valid_numbers)
        previous_fields = self._video_prompt_fields
        self._video_prompt_fields = {
            item.scene_number: _preserved_or_priority_video_prompt_field(
                item,
                previous_fields.get(item.scene_number, ""),
            )
            for item in self._items
        }
        self.endResetModel()

    def clear_selection(self) -> None:
        if not self._selected:
            return
        self._selected.clear()
        if self._items:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(len(self._items) - 1, 0),
                [self.SelectedRole],
            )

    def toggle_selection(self, scene_number: int) -> bool:
        row = self._row_for_scene(scene_number)
        if row is None:
            return False
        if scene_number in self._selected:
            self._selected.remove(scene_number)
        else:
            self._selected.add(scene_number)
        index = self.index(row, 0)
        self.dataChanged.emit(index, index, [self.SelectedRole])
        return True

    def select_only(self, scene_number: int) -> bool:
        row = self._row_for_scene(scene_number)
        if row is None:
            return False
        if self._selected == {scene_number}:
            return True
        self._selected = {scene_number}
        self.dataChanged.emit(
            self.index(0, 0),
            self.index(len(self._items) - 1, 0),
            [self.SelectedRole],
        )
        return True

    def is_selected(self, scene_number: int) -> bool:
        return scene_number in self._selected

    def contains(self, scene_number: int) -> bool:
        return self._row_for_scene(scene_number) is not None

    @property
    def selected_scene_numbers(self) -> tuple[int, ...]:
        return tuple(sorted(self._selected))

    def apply_prompt_fields(self, scene_number: int, fields: Mapping[str, str]) -> bool:
        row = self._row_for_scene(scene_number)
        if row is None:
            return False
        if "video_prompt" in fields and "video_prompt_field" in fields:
            self._video_prompt_fields[scene_number] = fields["video_prompt_field"]
        self._items[row] = _updated_prompt_item(self._items[row], fields)
        index = self.index(row, 0)
        self.dataChanged.emit(
            index,
            index,
            [
                self.ShotDescriptionRole,
                self.ImagePromptRole,
                self.VideoPromptRole,
                self.VideoPromptFieldRole,
            ],
        )
        return True

    def scene_map(self, scene_number: int) -> dict[str, Any]:
        row = self._row_for_scene(scene_number)
        if row is None:
            return {}
        item = self._items[row]
        video_prompt_field, video_prompt = self._selected_video_prompt(item)
        raw_ltx = item.raw_scene.get("ltx")
        ltx = raw_ltx if isinstance(raw_ltx, Mapping) else {}
        return {
            "sceneNumber": item.scene_number,
            "startSeconds": item.start_seconds,
            "endSeconds": item.end_seconds,
            "performanceState": item.performance_state,
            "status": item.status,
            "thumbnailUrl": self._thumbnail(item),
            "thumbnailPath": item.media.thumbnail_path or "",
            "workflowPath": item.media.workflow_path or "",
            "videoPath": item.media.video_path or "",
            "failureMessage": item.media.failure_message or "",
            "shotDescription": item.shot_description,
            "imagePrompt": item.image_prompt,
            "videoPrompt": video_prompt,
            "videoPromptField": video_prompt_field,
            "ltxPrompts": {
                field.value: str(ltx.get(field.value) or "")
                for field in SceneLtxPromptField
            },
            "referenceIds": list(item.reference_ids),
            "selected": item.scene_number in self._selected,
        }

    def _row_for_scene(self, scene_number: int) -> int | None:
        return next(
            (
                row
                for row, item in enumerate(self._items)
                if item.scene_number == scene_number
            ),
            None,
        )

    def _thumbnail(self, item: SceneWorkspaceItem) -> str:
        path = item.media.thumbnail_path
        if path:
            return self._thumbnail_url(path)
        video_path = item.media.video_path
        return self._video_thumbnail_url(video_path) if video_path else ""

    def _selected_video_prompt(self, item: SceneWorkspaceItem) -> tuple[str, str]:
        field = self._video_prompt_fields.get(item.scene_number, "")
        if not field:
            return "", item.video_prompt
        ltx = item.raw_scene.get("ltx")
        if not isinstance(ltx, Mapping):
            return field, ""
        return field, str(ltx.get(field) or "")


class SceneWorkspaceViewModel(QObject):
    scenesChanged = Signal()
    projectChanged = Signal()
    selectionChanged = Signal()
    stateChanged = Signal()
    inspectedSceneChanged = Signal()
    submittingChanged = Signal()

    def __init__(
        self,
        *,
        service: Any,
        studio_view_model: Any,
        thumbnail_url: Callable[[str, str], str] | None = None,
        video_thumbnail_url: Callable[[str, str], str] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._studio_view_model = studio_view_model
        self._current_project_id = ""
        self._revision = ""
        self._error = ""
        self._pending: dict[tuple[int, str], str] = {}
        self._conflicts: set[tuple[int, str]] = set()
        self._primary_scene_number: int | None = None
        self._baseline_items: tuple[SceneWorkspaceItem, ...] = ()
        self._submitting = False
        resolver = thumbnail_url or (lambda _project_id, path: _local_file_url(path))
        video_resolver = video_thumbnail_url or (lambda _project_id, _path: "")
        self._scenes = SceneListModel(
            thumbnail_url=lambda path: resolver(self._current_project_id, path),
            video_thumbnail_url=lambda path: video_resolver(
                self._current_project_id,
                path,
            ),
            parent=self,
        )
        project_signal = getattr(studio_view_model, "currentProjectChanged", None)
        if project_signal is not None:
            project_signal.connect(self._project_selected)

    @Property(QObject, constant=True)
    def scenes(self) -> SceneListModel:
        return self._scenes

    @Property(str, notify=projectChanged)
    def current_project_id(self) -> str:
        return self._current_project_id

    @Property(str, notify=stateChanged)
    def revision(self) -> str:
        return self._revision

    @Property("QVariantList", notify=selectionChanged)
    def selected_scene_numbers(self) -> list[int]:
        return list(self._scenes.selected_scene_numbers)

    @Property(bool, notify=stateChanged)
    def dirty(self) -> bool:
        return bool(self._pending)

    @Property(bool, notify=stateChanged)
    def conflict(self) -> bool:
        return bool(self._conflicts)

    @Property(str, notify=stateChanged)
    def error(self) -> str:
        return self._error

    @Property(bool, notify=submittingChanged)
    def submitting(self) -> bool:
        return self._submitting

    @Property("QVariantMap", notify=inspectedSceneChanged)
    def inspectedScene(self) -> dict[str, Any]:  # noqa: N802 - QML API
        if self._primary_scene_number is None:
            return {}
        return self._scenes.scene_map(self._primary_scene_number)

    @Property("QVariantMap", notify=inspectedSceneChanged)
    def currentScene(self) -> dict[str, Any]:  # noqa: N802 - QML API
        return self.inspectedScene

    @Slot(result=bool)
    def reload(self) -> bool:
        project_id = str(getattr(self._studio_view_model, "current_project_id", "") or "")
        project_changed = project_id != self._current_project_id
        if project_changed:
            self._current_project_id = project_id
            self._clear_workspace()
            self.projectChanged.emit()
        if not project_id:
            return True
        try:
            snapshot = self._service.load(project_id)
        except Exception as exc:  # noqa: BLE001 - Qt boundary
            self._set_state(error=str(exc))
            return False
        self._scenes.replace(snapshot.workspace.items)
        self._baseline_items = tuple(snapshot.workspace.items)
        self._pending.clear()
        self._conflicts.clear()
        self._revision = snapshot.revision
        self._error = ""
        if (
            self._primary_scene_number is not None
            and not self._scenes.is_selected(self._primary_scene_number)
        ):
            self._primary_scene_number = _first_selected(self._scenes)
        self.scenesChanged.emit()
        self.selectionChanged.emit()
        self.inspectedSceneChanged.emit()
        self.stateChanged.emit()
        return True

    @Slot(int, result=bool)
    def toggleSelection(self, scene_number: int) -> bool:  # noqa: N802 - QML API
        changed = self._scenes.toggle_selection(scene_number)
        if changed:
            if self._scenes.is_selected(scene_number):
                self._primary_scene_number = scene_number
            elif self._primary_scene_number == scene_number:
                self._primary_scene_number = _first_selected(self._scenes)
            self.selectionChanged.emit()
            self.inspectedSceneChanged.emit()
        return changed

    @Slot(int, bool, result=bool)
    def selectScene(self, scene_number: int, additive: bool) -> bool:  # noqa: N802 - QML API
        if additive:
            return self.toggleSelection(scene_number)
        changed = self._scenes.select_only(scene_number)
        if changed:
            self._primary_scene_number = scene_number
            self.selectionChanged.emit()
            self.inspectedSceneChanged.emit()
        return changed

    @Slot(int, "QVariantMap", str, result=bool)
    def savePromptFields(  # noqa: N802 - QML API
        self,
        scene_number: int,
        fields: Mapping[str, Any],
        ltx_prompt_field: str,
    ) -> bool:
        if not self._current_project_id:
            self._set_state(error="Select a project first")
            return False
        try:
            changes, local_fields, selected_field = _validated_prompt_patch(
                fields,
                ltx_prompt_field,
            )
        except (TypeError, ValueError) as exc:
            self._set_state(error=str(exc))
            return False
        if not self._scenes.contains(scene_number):
            self._set_state(error=f"Scene {scene_number} not found")
            return False

        patch_values = {
            (scene_number, field_name): value
            for field_name, value in changes.items()
        }
        self._scenes.apply_prompt_fields(scene_number, local_fields)
        self._pending.update(patch_values)
        self._set_state(error="", force=True)
        if self._primary_scene_number == scene_number:
            self.inspectedSceneChanged.emit()
        try:
            snapshot = self._service.patch_scene(
                project_id=self._current_project_id,
                scene_number=scene_number,
                changes=changes,
                expected_revision=self._revision,
                selected_ltx_prompt_field=selected_field,
            )
            for key, value in patch_values.items():
                if self._pending.get(key) == value:
                    self._pending.pop(key, None)
                    self._conflicts.discard(key)
            self._baseline_items = tuple(
                _updated_prompt_item(item, local_fields)
                if item.scene_number == scene_number else item
                for item in self._baseline_items
            )
            self._revision = str(snapshot.revision)
            self._set_state(error="", force=True)
            refresh_editor = getattr(
                self._studio_view_model,
                "refresh_render_plan_editor",
                None,
            )
            if callable(refresh_editor):
                refresh_editor()
            return True
        except SceneDocumentConflict as exc:
            self._conflicts.update(patch_values)
            self._set_state(error=str(exc), force=True)
            return False
        except Exception as exc:  # noqa: BLE001 - Qt boundary
            self._set_state(error=str(exc), force=True)
            return False

    @Slot(result=bool)
    def discardLocalEdits(self) -> bool:  # noqa: N802 - QML API
        self._scenes.replace(self._baseline_items)
        self._pending.clear()
        self._conflicts.clear()
        self._error = ""
        if (
            self._primary_scene_number is not None
            and not self._scenes.is_selected(self._primary_scene_number)
        ):
            self._primary_scene_number = _first_selected(self._scenes)
        self.scenesChanged.emit()
        self.selectionChanged.emit()
        self.inspectedSceneChanged.emit()
        self.stateChanged.emit()
        return True

    @Slot(str, result=bool)
    @Slot(str, int, result=bool)
    def startSelectedAction(  # noqa: N802 - QML API
        self,
        action: str,
        preview_stage: int = 0,
    ) -> bool:
        if self._submitting:
            return False
        self._submitting = True
        self.submittingChanged.emit()
        try:
            self._service.start_action(
                project_id=self._current_project_id,
                action=action,
                scene_numbers=self._scenes.selected_scene_numbers,
                preview_stage=preview_stage or None,
            )
            refresh_jobs = getattr(self._studio_view_model, "refresh_jobs", None)
            if callable(refresh_jobs):
                refresh_jobs()
            self._set_state(error="")
            return True
        except Exception as exc:  # noqa: BLE001 - Qt boundary
            self._set_state(error=str(exc))
            return False
        finally:
            self._submitting = False
            self.submittingChanged.emit()

    @Slot()
    def _project_selected(self) -> None:
        self.reload()

    def _set_state(
        self,
        *,
        revision: str | None = None,
        error: str | None = None,
        force: bool = False,
    ) -> None:
        changed = force
        for attribute, value in (
            ("_revision", revision),
            ("_error", error),
        ):
            if value is not None and getattr(self, attribute) != value:
                setattr(self, attribute, value)
                changed = True
        if changed:
            self.stateChanged.emit()

    def _clear_workspace(self) -> None:
        self._scenes.clear_selection()
        self._scenes.replace(())
        self._primary_scene_number = None
        self._baseline_items = ()
        self._pending.clear()
        self._conflicts.clear()
        self._revision = ""
        self._error = ""
        self.selectionChanged.emit()
        self.scenesChanged.emit()
        self.inspectedSceneChanged.emit()
        self.stateChanged.emit()


def _validated_prompt_patch(
    fields: Mapping[str, Any],
    ltx_prompt_field: str,
) -> tuple[dict[str, str], dict[str, str], SceneLtxPromptField | None]:
    aliases = {
        "shotDescription": "shot_description",
        "imagePrompt": "image_prompt",
        "videoPrompt": "video_prompt",
        "shot_description": "shot_description",
        "image_prompt": "image_prompt",
        "video_prompt": "video_prompt",
    }
    if not fields:
        raise ValueError("Scene patch requires at least one editable field")
    unknown = set(fields) - set(aliases)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Scene field is not editable: {names}")
    if any(not isinstance(value, str) for value in fields.values()):
        raise TypeError("Scene prompt fields must be text")
    local_fields: dict[str, str] = {}
    for name, value in fields.items():
        local_name = aliases[name]
        if local_name in local_fields:
            raise ValueError(f"Duplicate scene field: {local_name}")
        local_fields[local_name] = value
    changes: dict[str, str] = {}
    if "shot_description" in local_fields:
        changes["shot_description"] = local_fields["shot_description"]
    if "image_prompt" in local_fields:
        changes["z_image.prompt"] = local_fields["image_prompt"]
    selected_field = None
    if "video_prompt" in local_fields:
        selected_field = SceneLtxPromptField(ltx_prompt_field)
        changes[f"ltx.{selected_field.value}"] = local_fields["video_prompt"]
        local_fields["video_prompt_field"] = selected_field.value
    return changes, local_fields, selected_field


def _updated_prompt_item(
    item: SceneWorkspaceItem,
    fields: Mapping[str, str],
) -> SceneWorkspaceItem:
    raw_scene = item.raw_scene
    if "video_prompt" in fields and "video_prompt_field" in fields:
        raw_ltx = raw_scene.get("ltx")
        ltx = dict(raw_ltx) if isinstance(raw_ltx, Mapping) else {}
        ltx[fields["video_prompt_field"]] = fields["video_prompt"]
        raw_scene["ltx"] = ltx
    return replace(
        item,
        shot_description=fields.get("shot_description", item.shot_description),
        image_prompt=fields.get("image_prompt", item.image_prompt),
        video_prompt=fields.get("video_prompt", item.video_prompt),
        _raw_scene=raw_scene,
    )


def _priority_video_prompt_field(item: SceneWorkspaceItem) -> str:
    ltx = item.raw_scene.get("ltx")
    if not isinstance(ltx, Mapping):
        return ""
    for field in SceneLtxPromptField:
        if ltx.get(field.value):
            return field.value
    return ""


def _preserved_or_priority_video_prompt_field(
    item: SceneWorkspaceItem,
    previous_field: str,
) -> str:
    ltx = item.raw_scene.get("ltx")
    if isinstance(ltx, Mapping) and previous_field in ltx:
        return previous_field
    return _priority_video_prompt_field(item)


def _first_selected(model: SceneListModel) -> int | None:
    selected = model.selected_scene_numbers
    return selected[0] if selected else None


def _local_file_url(path: str) -> str:
    url = QUrl(path)
    if url.isLocalFile():
        return url.toString()
    return QUrl.fromLocalFile(str(Path(path).resolve())).toString()
