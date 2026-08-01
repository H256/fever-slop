"""Qt view model for prompt revision history and rebuild preview."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

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

from feverslop.domain.rebuild_policy import ArtifactKind
from feverslop.domain.prompt_revisions import PromptField
from feverslop.studio.rebuild_service import PromptSaveConflict, RebuildService


@dataclass(frozen=True)
class RevisionEntry:
    revision_id: str
    value: str
    at: str
    restored_from: str | None
    parent_id: str | None


class RevisionListModel(QAbstractListModel):
    """List model presenting individual prompt revisions."""

    IdRole = Qt.ItemDataRole.UserRole + 1
    ValueRole = IdRole + 1
    TimestampRole = IdRole + 2
    RestoredFromRole = IdRole + 3
    ParentIdRole = IdRole + 4
    DiffRole = IdRole + 5
    IsCurrentRole = IdRole + 6

    _ROLE_NAMES = {
        IdRole: b"id",
        ValueRole: b"value",
        TimestampRole: b"timestamp",
        RestoredFromRole: b"restoredFrom",
        ParentIdRole: b"parentId",
        DiffRole: b"diff",
        IsCurrentRole: b"isCurrent",
    }

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._entries: list[RevisionEntry] = []
        self._last_value: str = ""

    def roleNames(self) -> dict[int, bytes]:  # noqa: N802 - Qt API
        return dict(self._ROLE_NAMES)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._entries)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or not 0 <= index.row() < len(self._entries):
            return None
        entry = self._entries[index.row()]
        if role == self.ValueRole:
            return entry.value
        if role == self.IdRole:
            return entry.revision_id
        if role == self.TimestampRole:
            return entry.at
        if role == self.RestoredFromRole:
            return entry.restored_from or ""
        if role == self.ParentIdRole:
            return entry.parent_id or ""
        if role == self.IsCurrentRole:
            return entry.value == self._last_value
        if role == self.DiffRole:
            return _compute_diff(entry.value, self._last_value if entry.value != self._last_value else "")
        return None

    def refresh(self, entries: list[RevisionEntry]) -> None:
        self.beginResetModel()
        if entries:
            self._last_value = entries[-1].value
        self._entries = list(reversed(entries))
        self.endResetModel()


class RebuildViewModel(QObject):
    revisionsChanged = Signal()
    previewChanged = Signal()
    stateChanged = Signal()

    def __init__(
        self,
        *,
        service: RebuildService,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._project_id = ""
        self._scene_number = 0
        self._field = PromptField.Z_IMAGE_PROMPT
        self._error = ""
        self._selected_revision_id = ""
        self._revisions = RevisionListModel(parent=self)

    @Property(RevisionListModel, constant=True)
    def revisions(self) -> RevisionListModel:
        return self._revisions

    @Property(str, notify=stateChanged)
    def selected_revision_id(self) -> str:
        return self._selected_revision_id

    @selected_revision_id.setter  # type: ignore[assignment]
    def selected_revision_id(self, value: str) -> None:
        self._selected_revision_id = value
        self.stateChanged.emit()

    @Property(str, notify=stateChanged)
    def error(self) -> str:
        return self._error

    @Slot(int, str, result=bool)
    def loadRevisions(self, scene_number: int, field_value: str) -> bool:  # noqa: N802
        self._scene_number = scene_number
        try:
            self._field = PromptField(field_value)
        except ValueError:
            self._field = PromptField.Z_IMAGE_PROMPT
        try:
            result = self._service.get_history(
                project_id=self._project_id,
                scene_number=scene_number,
                field=self._field,
            )
            entries = [
                RevisionEntry(
                    revision_id=rev.id,
                    value=rev.value,
                    at=_format_timestamp(rev.created_at),
                    restored_from=rev.restored_from,
                    parent_id=rev.parent_id,
                )
                for rev in result.history.revisions
            ]
            self._revisions.refresh(entries)
            self._selected_revision_id = ""
            self._clear_error()
            self.revisionsChanged.emit()
            return True
        except Exception as exc:  # noqa: BLE001 - Qt boundary
            self._set_error(str(exc))
            return False

    @Slot(result=bool)
    def restoreSelected(self) -> bool:  # noqa: N802 - QML API
        if not self._selected_revision_id:
            self._set_error("No revision selected")
            return False
        try:
            self._service.restore_revision(
                project_id=self._project_id,
                scene_number=self._scene_number,
                field=self._field,
                revision_id=self._selected_revision_id,
            )
            self._clear_error()
            self.loadRevisions(self._scene_number, self._field.value)
            return True
        except PromptSaveConflict as exc:
            self._set_error(str(exc))
            return False
        except Exception as exc:  # noqa: BLE001 - Qt boundary
            self._set_error(str(exc))
            return False

    @Slot(str, result=str)
    def getLocalFileUrl(self, path: str) -> str:  # noqa: N802 - QML API
        url = QUrl(path)
        if url.isLocalFile():
            return url.toString()
        return QUrl.fromLocalFile(str(Path(path).resolve())).toString()

    @Slot(list, result=str)
    def artifactKindsToAction(self, kinds: list) -> str:  # noqa: N802 - QML API
        return artifact_kinds_to_action(kinds)

    def set_project_id(self, project_id: str) -> None:
        self._project_id = project_id

    def _set_error(self, message: str) -> None:
        self._error = message
        self.stateChanged.emit()

    def _clear_error(self) -> None:
        self._error = ""
        self.stateChanged.emit()


def _format_timestamp(at: datetime | str) -> str:
    if isinstance(at, datetime):
        return at.isoformat()
    return at


def _compute_diff(current: str, previous: str) -> str:
    import difflib
    if not previous:
        return f"Initial value: {current}"
    diff = difflib.unified_diff(
        previous.splitlines(keepends=True),
        current.splitlines(keepends=True),
        fromfile="previous",
        tofile="current",
    )
    return "".join(diff)


# Map artifact kinds to job action names for rebuild execution.
_ARTIFACT_KIND_TO_ACTION: dict[str, str] = {
    ArtifactKind.RENDER_PLAN.value: "rebuild-plan-timeline",
    ArtifactKind.PROMPT_GENERATION.value: "rebuild-plan-timeline",
    ArtifactKind.SCENE_RENDER.value: "ltx-render-scenes",
    ArtifactKind.FINAL_VIDEO.value: "final-concat",
    ArtifactKind.PREPARED_WORKFLOW.value: "rebuild-plan",
    ArtifactKind.REFERENCE_SHEETS.value: "rebuild-plan",
    ArtifactKind.SCENE_STORYBOARD.value: "storyboard",
    ArtifactKind.REVIEW_ORDERING.value: "rebuild-plan-timeline",
}


def artifact_kinds_to_action(kinds: Sequence[str | object]) -> str:
    """Convert artifact kind names/values to the corresponding job action name.

    Prefers timeline-aware rebuild for multi-artifact scenarios.
    """
    kind_values = set()
    for kind in kinds:
        if isinstance(kind, ArtifactKind):
            kind_values.add(kind.value)
        elif isinstance(kind, str):
            kind_values.add(kind)

    if not kind_values:
        return "rebuild-plan-timeline"

    # If there are prompt/render-plan changes, use the timeline-aware rebuild
    if ArtifactKind.RENDER_PLAN.value in kind_values or ArtifactKind.PROMPT_GENERATION.value in kind_values:
        return "rebuild-plan-timeline"

    # First try direct mapping, then default to timeline rebuild
    for kind_val in sorted(kind_values):
        action = _ARTIFACT_KIND_TO_ACTION.get(kind_val)
        if action:
            return action

    return "rebuild-plan-timeline"
