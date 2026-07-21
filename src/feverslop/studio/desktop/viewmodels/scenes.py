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
    ReferenceIdsRole = SceneNumberRole + 9
    SelectedRole = SceneNumberRole + 10

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
        ReferenceIdsRole: b"referenceIds",
        SelectedRole: b"selected",
    }

    def __init__(
        self,
        *,
        thumbnail_url: Callable[[str], str] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._items: list[SceneWorkspaceItem] = []
        self._selected: set[int] = set()
        self._thumbnail_url = thumbnail_url or _local_file_url

    def roleNames(self) -> dict[int, bytes]:  # noqa: N802 - Qt API
        return dict(self._ROLE_NAMES)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802 - Qt API
        return 0 if parent.isValid() else len(self._items)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or not 0 <= index.row() < len(self._items):
            return None
        item = self._items[index.row()]
        values = {
            self.SceneNumberRole: item.scene_number,
            self.StartSecondsRole: item.start_seconds,
            self.EndSecondsRole: item.end_seconds,
            self.PerformanceStateRole: item.performance_state,
            self.StatusRole: item.status,
            self.ThumbnailUrlRole: self._thumbnail(item),
            self.ShotDescriptionRole: item.shot_description,
            self.ImagePromptRole: item.image_prompt,
            self.VideoPromptRole: item.video_prompt,
            self.ReferenceIdsRole: list(item.reference_ids),
            self.SelectedRole: item.scene_number in self._selected,
        }
        return values.get(role)

    def replace(self, items: Iterable[SceneWorkspaceItem]) -> None:
        self.beginResetModel()
        self._items = list(items)
        valid_numbers = {item.scene_number for item in self._items}
        self._selected.intersection_update(valid_numbers)
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

    @property
    def selected_scene_numbers(self) -> tuple[int, ...]:
        return tuple(sorted(self._selected))

    def apply_prompt_fields(self, scene_number: int, fields: Mapping[str, str]) -> bool:
        row = self._row_for_scene(scene_number)
        if row is None:
            return False
        item = self._items[row]
        self._items[row] = replace(
            item,
            shot_description=fields.get("shot_description", item.shot_description),
            image_prompt=fields.get("image_prompt", item.image_prompt),
            video_prompt=fields.get("video_prompt", item.video_prompt),
        )
        index = self.index(row, 0)
        self.dataChanged.emit(
            index,
            index,
            [self.ShotDescriptionRole, self.ImagePromptRole, self.VideoPromptRole],
        )
        return True

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
        return self._thumbnail_url(path) if path else ""


class SceneWorkspaceViewModel(QObject):
    scenesChanged = Signal()
    projectChanged = Signal()
    selectionChanged = Signal()
    stateChanged = Signal()

    def __init__(
        self,
        *,
        service: Any,
        studio_view_model: Any,
        thumbnail_url: Callable[[str, str], str] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._studio_view_model = studio_view_model
        self._current_project_id = ""
        self._revision = ""
        self._dirty = False
        self._conflict = False
        self._error = ""
        resolver = thumbnail_url or (lambda _project_id, path: _local_file_url(path))
        self._scenes = SceneListModel(
            thumbnail_url=lambda path: resolver(self._current_project_id, path),
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
        return self._dirty

    @Property(bool, notify=stateChanged)
    def conflict(self) -> bool:
        return self._conflict

    @Property(str, notify=stateChanged)
    def error(self) -> str:
        return self._error

    @Slot(result=bool)
    def reload(self) -> bool:
        project_id = str(getattr(self._studio_view_model, "current_project_id", "") or "")
        project_changed = project_id != self._current_project_id
        self._current_project_id = project_id
        self._scenes.clear_selection()
        self.selectionChanged.emit()
        self._scenes.replace(())
        self.scenesChanged.emit()
        self._set_state(revision="", dirty=False, conflict=False, error="")
        if project_changed:
            self.projectChanged.emit()
        if not project_id:
            return True
        try:
            snapshot = self._service.load(project_id)
            self._scenes.replace(snapshot.workspace.items)
            self.scenesChanged.emit()
            self._set_state(revision=snapshot.revision)
            return True
        except Exception as exc:  # noqa: BLE001 - Qt boundary
            self._set_state(error=str(exc))
            return False

    @Slot(int, result=bool)
    def toggleSelection(self, scene_number: int) -> bool:  # noqa: N802 - QML API
        changed = self._scenes.toggle_selection(scene_number)
        if changed:
            self.selectionChanged.emit()
        return changed

    @Slot(int, "QVariantMap", str, result=bool)
    def savePromptFields(  # noqa: N802 - QML API
        self,
        scene_number: int,
        fields: Mapping[str, Any],
        ltx_prompt_field: str,
    ) -> bool:
        local_fields = _local_prompt_fields(fields)
        if not self._scenes.apply_prompt_fields(scene_number, local_fields):
            self._set_state(dirty=True, conflict=False, error=f"Scene {scene_number} not found")
            return False
        self._set_state(dirty=True, conflict=False, error="")
        try:
            changes, selected_field = _prompt_changes(local_fields, ltx_prompt_field)
            snapshot = self._service.patch_scene(
                project_id=self._current_project_id,
                scene_number=scene_number,
                changes=changes,
                expected_revision=self._revision,
                selected_ltx_prompt_field=selected_field,
            )
            self._set_state(
                revision=str(snapshot.revision),
                dirty=False,
                conflict=False,
                error="",
            )
            return True
        except SceneDocumentConflict as exc:
            self._set_state(dirty=True, conflict=True, error=str(exc))
            return False
        except Exception as exc:  # noqa: BLE001 - Qt boundary
            self._set_state(dirty=True, conflict=False, error=str(exc))
            return False

    @Slot(str, result=bool)
    @Slot(str, int, result=bool)
    def startSelectedAction(  # noqa: N802 - QML API
        self,
        action: str,
        preview_stage: int = 0,
    ) -> bool:
        try:
            self._service.start_action(
                project_id=self._current_project_id,
                action=action,
                scene_numbers=self._scenes.selected_scene_numbers,
                preview_stage=preview_stage or None,
            )
            self._set_state(error="")
            return True
        except Exception as exc:  # noqa: BLE001 - Qt boundary
            self._set_state(error=str(exc))
            return False

    @Slot()
    def _project_selected(self) -> None:
        self.reload()

    def _set_state(
        self,
        *,
        revision: str | None = None,
        dirty: bool | None = None,
        conflict: bool | None = None,
        error: str | None = None,
    ) -> None:
        changed = False
        for attribute, value in (
            ("_revision", revision),
            ("_dirty", dirty),
            ("_conflict", conflict),
            ("_error", error),
        ):
            if value is not None and getattr(self, attribute) != value:
                setattr(self, attribute, value)
                changed = True
        if changed:
            self.stateChanged.emit()


def _local_prompt_fields(fields: Mapping[str, Any]) -> dict[str, str]:
    aliases = {
        "shotDescription": "shot_description",
        "imagePrompt": "image_prompt",
        "videoPrompt": "video_prompt",
        "shot_description": "shot_description",
        "image_prompt": "image_prompt",
        "video_prompt": "video_prompt",
    }
    return {
        aliases[name]: str(value)
        for name, value in fields.items()
        if name in aliases
    }


def _prompt_changes(
    fields: Mapping[str, str],
    ltx_prompt_field: str,
) -> tuple[dict[str, str], SceneLtxPromptField | None]:
    changes: dict[str, str] = {}
    if "shot_description" in fields:
        changes["shot_description"] = fields["shot_description"]
    if "image_prompt" in fields:
        changes["z_image.prompt"] = fields["image_prompt"]
    selected_field = None
    if "video_prompt" in fields:
        selected_field = SceneLtxPromptField(ltx_prompt_field)
        changes[f"ltx.{selected_field.value}"] = fields["video_prompt"]
    return changes, selected_field


def _local_file_url(path: str) -> str:
    url = QUrl(path)
    if url.isLocalFile():
        return url.toString()
    return QUrl.fromLocalFile(str(Path(path).resolve())).toString()
