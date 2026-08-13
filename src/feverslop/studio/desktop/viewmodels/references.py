from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QObject,
    Property,
    Qt,
    Signal,
    Slot,
)

from feverslop.domain.reference_workspace import (
    ReferenceAsset,
    ReferenceKind,
    SceneReferenceAssignment,
)
from feverslop.studio.reference_workspace_service import (
    CommandResult,
    GenerationCommand,
    ReferenceWorkspaceService,
)


class ReferenceListModel(QAbstractListModel):
    """QML-exposed list of reference assets with filtering."""

    IdRole = Qt.ItemDataRole.UserRole + 1
    KindRole = IdRole + 1
    LabelRole = IdRole + 2
    PathRole = IdRole + 3
    WidthRole = IdRole + 4
    HeightRole = IdRole + 5
    ExistsRole = IdRole + 6
    StaleRole = IdRole + 7
    SourceRole = IdRole + 8
    ThumbnailUrlRole = IdRole + 9

    _ROLE_NAMES: dict[int, bytes] = {
        IdRole: b"id",
        KindRole: b"kind",
        LabelRole: b"label",
        PathRole: b"path",
        WidthRole: b"width",
        HeightRole: b"height",
        ExistsRole: b"exists",
        StaleRole: b"stale",
        SourceRole: b"source",
        ThumbnailUrlRole: b"thumbnailUrl",
    }

    dataChanged = Signal()

    def __init__(
        self,
        *,
        media_url: Callable[[str], str] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._items: list[ReferenceAsset] = []
        self._media_url = media_url or (lambda p: "")
        self._selection: str = ""

    def roleNames(self) -> dict[int, bytes]:  # noqa: N802
        return dict(self._ROLE_NAMES)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._items)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:  # noqa: N802
        if not index.isValid() or not 0 <= index.row() < len(self._items):
            return None
        asset = self._items[index.row()]
        values = {
            self.IdRole: asset.id,
            self.KindRole: asset.kind.value,
            self.LabelRole: asset.label or asset.id,
            self.PathRole: asset.path,
            self.WidthRole: asset.width,
            self.HeightRole: asset.height,
            self.ExistsRole: asset.exists,
            self.StaleRole: asset.stale,
            self.SourceRole: (asset.provenance.source if asset.provenance else ""),
            self.ThumbnailUrlRole: self._media_url(asset.path) if asset.exists else "",
        }
        return values.get(role)

    def replace(self, assets: tuple[ReferenceAsset, ...]) -> None:
        self.beginResetModel()
        self._items = list(assets)
        self.endResetModel()

    def selected_id(self) -> str:
        return self._selection or ""

    def set_selected_id(self, id: str) -> None:  # noqa: A002
        self._selection = id
        self.dataChanged.emit()


class SceneAssignmentListModel(QAbstractListModel):
    """QML-exposed list of per-scene reference assignments."""

    SceneNumberRole = Qt.ItemDataRole.UserRole + 1
    ActorIdsRole = SceneNumberRole + 1
    LocationIdsRole = SceneNumberRole + 2
    BackgroundIdsRole = SceneNumberRole + 3
    StyleIdsRole = SceneNumberRole + 4
    ActorLookIdsRole = SceneNumberRole + 5

    _ROLE_NAMES: dict[int, bytes] = {
        SceneNumberRole: b"sceneNumber",
        ActorIdsRole: b"actorIds",
        LocationIdsRole: b"locationIds",
        BackgroundIdsRole: b"backgroundIds",
        StyleIdsRole: b"styleIds",
        ActorLookIdsRole: b"actorLookIds",
    }

    def __init__(self, *, parent: QObject | None = None) -> None:  # noqa: A002
        super().__init__(parent)
        self._items: list[dict[str, Any]] = []

    def roleNames(self) -> dict[int, bytes]:  # noqa: N802
        return dict(self._ROLE_NAMES)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._items)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:  # noqa: N802
        if not index.isValid() or not 0 <= index.row() < len(self._items):
            return None
        item = self._items[index.row()]
        actor_look_ids = item.get("actor_look_ids") or {}
        values = {
            self.SceneNumberRole: item.get("scene_number"),
            self.ActorIdsRole: list(item.get("actor_ids") or []),
            self.LocationIdsRole: list(item.get("location_ids") or []),
            self.BackgroundIdsRole: list(item.get("background_ids") or []),
            self.StyleIdsRole: list(item.get("style_ids") or []),
            self.ActorLookIdsRole: dict(actor_look_ids),
        }
        return values.get(role)

    def replace(self, assignments: tuple[SceneReferenceAssignment, ...]) -> None:
        self.beginResetModel()
        self._items = [_assignment_to_dict(a) for a in sorted(assignments, key=lambda a: a.scene_number)]
        self.endResetModel()


class ReferenceWorkspaceViewModel(QObject):
    """Exposes reference workspace operations to QML."""

    libraryChanged = Signal()
    assignmentsChanged = Signal()

    currentProjectChanged = Signal()
    selectedAssetChanged = Signal()
    statusMessageChanged = Signal()
    errorMessageChanged = Signal()
    revisionChanged = Signal()

    def __init__(
        self,
        *,
        service: ReferenceWorkspaceService,
        media_url: Callable[[str], str] | None = None,
        project_root: str = "",
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._project_root = project_root
        if media_url:
            self._media_url = media_url
        elif project_root:
            self._media_url = self._resolve_media_url
        else:
            self._media_url = lambda p: ""
        self._library_model = ReferenceListModel(media_url=self._media_url, parent=self)
        self._assignments_model = SceneAssignmentListModel(parent=self)
        self._current_project: str = ""
        self._selected_asset: str = ""
        self._status_message: str = ""
        self._error_message: str = ""
        self._filter_kind: str = ""
        self._stale_only: bool = False
        self._missing_only: bool = False
        self._preview_result: CommandResult | None = None

    # -- QML properties --

    def _resolve_media_url(self, path: str) -> str:  # type: ignore[misc]
        from pathlib import Path
        from PySide6.QtCore import QUrl

        p = Path(path)
        if not p.is_absolute() and self._project_root:
            p = Path(self._project_root) / p
        return QUrl.fromLocalFile(str(p)).toString()

    @Property("QVariant", notify=libraryChanged)
    def library_model(self) -> ReferenceListModel:
        return self._library_model

    @Property("QVariant", notify=assignmentsChanged)
    def assignments_model(self) -> SceneAssignmentListModel:
        return self._assignments_model

    @Property(str, notify=currentProjectChanged)
    def current_project(self) -> str:
        return self._current_project

    @Property(str, notify=selectedAssetChanged)
    def selected_asset(self) -> str:
        return self._selected_asset

    @Property(str, notify=statusMessageChanged)
    def status_message(self) -> str:
        return self._status_message

    @Property(str, notify=errorMessageChanged)
    def error_message(self) -> str:
        return self._error_message

    @Property(str, notify=revisionChanged)
    def revision(self) -> str:
        snap = self._service.load_library(self._current_project)
        return snap.revision

    @Property(bool, notify=libraryChanged)
    def has_projects(self) -> bool:
        return self._current_project != ""

    # -- QML slots --

    @Slot(str, result=bool)
    def set_project(self, project_id: str) -> bool:
        if project_id == self._current_project:
            return True
        self._current_project = project_id
        self._service.set_project(project_id)
        self._refresh_library()
        self._refresh_assignments()
        self.currentProjectChanged.emit()
        return True

    @Slot(str)
    def set_filter_kind(self, kind: str) -> None:
        self._filter_kind = kind
        self._refresh_library()

    @Slot(bool)
    def set_stale_only(self, value: bool) -> None:
        self._stale_only = value
        self._refresh_library()

    @Slot(bool)
    def set_missing_only(self, value: bool) -> None:
        self._missing_only = value
        self._refresh_library()

    @Slot(str)
    def select_asset(self, asset_id: str) -> None:
        self._selected_asset = asset_id
        self.selectedAssetChanged.emit()

    @Slot(result=str)
    def selected_asset_path(self) -> str:
        if not self._selected_asset or not self._current_project:
            return ""
        asset = self._service.get_asset(self._current_project, self._selected_asset)
        if asset and asset.exists:
            return self._media_url(asset.path)
        return ""

    @Slot(result="QVariant")
    def selected_asset_info(self) -> dict[str, Any]:
        if not self._selected_asset or not self._current_project:
            return {}
        asset = self._service.get_asset(self._current_project, self._selected_asset)
        if not asset:
            return {}
        return _asset_to_dict(asset)

    @Slot(result="QVariantList")
    def scenes_for_asset(self) -> list[int]:
        if not self._selected_asset or not self._current_project:
            return []
        scenes = self._service.scenes_using_asset(self._current_project, self._selected_asset)
        return list(scenes)

    @Slot(int, "QVariantList", "QVariantList", "QVariantList", result="QVariant")
    def preview_assignment(self, scene_number: int, actor_ids: list, location_ids: list, background_ids: list) -> dict:
        result = self._service.preview_assignment(
            self._current_project,
            scene_number,
            actor_ids=tuple(actor_ids),
            location_ids=tuple(location_ids),
            background_ids=tuple(background_ids),
        )
        self._error_message = "" if result.success else (result.error.message if result.error else "")
        self.errorMessageChanged.emit()
        if result.success:
            self._preview_result = result
        return _command_result_to_dict(result)

    @Slot("QVariantList", result=str)
    def save_assignments(self, assignments_data: list) -> str:
        if not self._current_project:
            return ""
        old_snap = self._service.load_library(self._current_project)
        result = self._service.save_assignments(
            self._current_project,
            tuple(assignments_data),
            old_snap.revision,
            old_snapshot=old_snap,
        )
        if result.success:
            self._refresh_assignments()
            self._status_message = f"Saved revision {result.data.get('new_revision', '?')}"
            self.statusMessageChanged.emit()
            return result.data.get("new_revision", "")
        self._error_message = result.error.message if result.error else "Save failed"
        self.errorMessageChanged.emit()
        return ""

    @Slot(result="QVariantList")
    def collect_assignments(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for row in range(self._assignments_model.rowCount()):
            idx = self._assignments_model.index(row)
            item = {"scene_number": 0, "actor_ids": [], "location_ids": [], "background_ids": [], "style_ids": [], "actor_look_ids": {}}
            for role, key in [
                (self._assignments_model.SceneNumberRole, "scene_number"),
                (self._assignments_model.ActorIdsRole, "actor_ids"),
                (self._assignments_model.LocationIdsRole, "location_ids"),
                (self._assignments_model.BackgroundIdsRole, "background_ids"),
                (self._assignments_model.StyleIdsRole, "style_ids"),
                (self._assignments_model.ActorLookIdsRole, "actor_look_ids"),
            ]:
                val = self._assignments_model.data(idx, role)
                if val is not None:
                    item[key] = val
            result.append(item)
        return result

    @Slot(str, dict, result="QVariant")
    def queue_generation(self, action: str, params: dict) -> dict:
        cmd = GenerationCommand(
            action=action,
            scene_number=params.get("scene_number"),
            reference_ids=tuple(params.get("reference_ids") or []),
            actor_ids=tuple(params.get("actor_ids") or []),
            location_ids=tuple(params.get("location_ids") or []),
            background_ids=tuple(params.get("background_ids") or []),
        )
        result = self._service.queue_generation(self._current_project, cmd)
        self._error_message = "" if result.success else (result.error.message if result.error else "")
        self.errorMessageChanged.emit()
        return _command_result_to_dict(result)

    # -- Internal --

    def _refresh_library(self) -> None:
        if not self._current_project:
            self._library_model.replace(())
            return
        kinds: list[ReferenceKind] | None = None
        if self._filter_kind:
            try:
                kinds = [ReferenceKind(self._filter_kind)]
            except ValueError:
                kinds = None
        snap = self._service.load_library(self._current_project)
        assets = snap.filter_assets(
            kinds=kinds or list(ReferenceKind),
            stale_only=self._stale_only,
            missing_only=self._missing_only,
        )
        self._library_model.replace(assets)
        self.libraryChanged.emit()

    def _refresh_assignments(self) -> None:
        if not self._current_project:
            self._assignments_model.replace(())
            return
        snap = self._service.load_library(self._current_project)
        self._assignments_model.replace(snap.assignments)
        self.assignmentsChanged.emit()


# -- Helpers --


def _assignment_to_dict(a: SceneReferenceAssignment) -> dict[str, Any]:
    return {
        "scene_number": a.scene_number,
        "actor_ids": list(a.actor_ids),
        "location_ids": list(a.location_ids),
        "background_ids": list(a.background_ids),
        "style_ids": list(a.style_ids),
        "actor_look_ids": dict(a.actor_look_ids or {}),
    }


def _asset_to_dict(a: ReferenceAsset) -> dict[str, Any]:
    prov = a.provenance
    return {
        "id": a.id,
        "kind": a.kind.value,
        "label": a.label or a.id,
        "path": a.path,
        "width": a.width,
        "height": a.height,
        "exists": a.exists,
        "stale": a.stale,
        "generation_state": a.generation_state,
        "source": prov.source if prov else "",
        "generated_by": prov.generated_by if prov else "",
    }


def _command_result_to_dict(r: CommandResult) -> dict[str, Any]:
    result: dict[str, Any] = {"success": r.success, "data": r.data}
    if r.error:
        result["error_code"] = r.error.code
        result["error_message"] = r.error.message
    return result
