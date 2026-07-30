"""QML-compatible view model for the timeline editing studio.

Wraps :class:`TimelineStudioService` behind Qt signals/properties so QML can
bind to timeline data and trigger edits via slots.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Property, QObject, Signal, Slot

from feverslop.ports.timeline_documents import AffectedArtifacts


class TimelineStudioViewModel(QObject):
    """QML viewmodel backed by :class:`TimelineStudioService`.

    All mutating commands push snapshots to the service's undo stack.
    State changes are propagated via *Changed* signals.
    """

    segmentsChanged = Signal()
    boundariesChanged = Signal()
    beatsChanged = Signal()
    statusChanged = Signal()
    errorChanged = Signal()
    undoRedoChanged = Signal()
    hasChangesChanged = Signal()

    def __init__(
        self,
        *,
        service: Any,
        project_dir: str = "",
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._project_dir = project_dir
        self._status = "idle"
        self._error = ""
        self._has_changes = False
        self._rebuilding = False
        self._segments: list[dict[str, Any]] = []
        self._boundaries: list[dict[str, Any]] = []
        self._beats: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Status properties
    # ------------------------------------------------------------------

    @Property(str, notify=statusChanged)
    def status(self) -> str:
        return self._status

    @Property(str, notify=errorChanged)
    def error(self) -> str:
        return self._error

    @Property(bool, notify=hasChangesChanged)
    def hasChanges(self) -> bool:  # noqa: N802
        return self._has_changes

    @Property(bool, notify=undoRedoChanged)
    def canUndo(self) -> bool:  # noqa: N802
        try:
            return self._service.can_undo()
        except Exception:
            return False

    @Property(bool, notify=undoRedoChanged)
    def canRedo(self) -> bool:  # noqa: N802
        try:
            return self._service.can_redo()
        except Exception:
            return False

    @Property(bool, notify=statusChanged)
    def rebuilding(self) -> bool:
        return self._rebuilding

    # ------------------------------------------------------------------
    # Observable data
    # ------------------------------------------------------------------

    @Property("QVariantList", notify=segmentsChanged)
    def segments(self) -> list[dict[str, Any]]:
        return list(self._segments)

    @Property("QVariantList", notify=boundariesChanged)
    def boundaries(self) -> list[dict[str, Any]]:
        return list(self._boundaries)

    @Property("QVariantList", notify=beatsChanged)
    def beats(self) -> list[dict[str, Any]]:
        return list(self._beats)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    @Slot(result=bool)
    def loadProject(self) -> bool:  # noqa: N802
        try:
            self._set_status("loading", error="")
            self._service.load()

            # Materialize observable lists from current snapshot
            snapshot = self._service.current_snapshot()
            self._segments = [
                {
                    "start": s.start,
                    "end": s.end,
                    "kind": s.kind,
                    "text": s.text,
                    "lyrics_line": s.lyrics_line or "",
                    "notes": s.notes or "",
                    "is_draft": s.is_draft,
                }
                for s in snapshot.segments
            ]
            self._boundaries = [
                {
                    "start": b.start,
                    "end": b.end,
                    "reason": b.reason,
                }
                for b in snapshot.scene_boundaries
            ]
            self._beats = [
                {
                    "time_s": m.time_s,
                    "label": m.label,
                    "confidence": m.confidence,
                }
                for m in snapshot.beat_markers
            ]
            self._has_changes = False
            self._set_status("loaded", error="")
            self.segmentsChanged.emit()
            self.boundariesChanged.emit()
            self.beatsChanged.emit()
            self.hasChangesChanged.emit()
            self.undoRedoChanged.emit()
            return True
        except Exception as exc:  # noqa: BLE001
            self._set_status("error", error=str(exc))
            return False

    @Slot(result=bool)
    def save(self) -> bool:
        try:
            self._service.save()
            self._has_changes = False
            self._set_status("saved", error="")
            return True
        except Exception as exc:  # noqa: BLE001
            self._set_status("error", error=str(exc))
            return False

    @Slot(int, float, float, str, str, result=bool)
    def editSegment(  # noqa: N802
        self,
        index: int,
        startDelta: float,
        endDelta: float,
        lyrics: str,
        notes: str,
    ) -> bool:
        try:
            # Normalize empty strings to None for domain
            lyrics_val: str | None = lyrics if lyrics else None
            notes_val: str | None = notes if notes else None

            self._service.edit_segment(
                index=index,
                start_delta=startDelta,
                end_delta=endDelta,
                lyrics=lyrics_val,
                notes=notes_val,
            )
            self._refresh_segments()
            self._has_changes = True
            self._set_status("edited", error="")
            self.undoRedoChanged.emit()
            return True
        except Exception as exc:  # noqa: BLE001
            self._set_status("error", error=str(exc))
            return False

    @Slot(int, float, result=bool)
    def splitSegment(self, index: int, at: float) -> bool:  # noqa: N802
        try:
            self._service.split_segment(index, at)
            self._refresh_segments()
            self._has_changes = True
            self._set_status("edited", error="")
            self.undoRedoChanged.emit()
            return True
        except Exception as exc:  # noqa: BLE001
            self._set_status("error", error=str(exc))
            return False

    @Slot(int, int, result=bool)
    def mergeSegments(self, index: int, count: int) -> bool:  # noqa: N802
        try:
            self._service.merge_segments(index, count)
            self._refresh_segments()
            self._has_changes = True
            self._set_status("edited", error="")
            self.undoRedoChanged.emit()
            return True
        except Exception as exc:  # noqa: BLE001
            self._set_status("error", error=str(exc))
            return False

    @Slot(float, float, str, result=bool)
    def addSceneBoundary(self, start: float, end: float, reason: str) -> bool:  # noqa: N802
        try:
            self._service.add_scene_boundary(start, end, reason)
            self._refresh_boundaries()
            self._has_changes = True
            self._set_status("edited", error="")
            self.undoRedoChanged.emit()
            return True
        except Exception as exc:  # noqa: BLE001
            self._set_status("error", error=str(exc))
            return False

    @Slot(float, str, float, result=bool)
    def addBeat(self, t: float, label: str, confidence: float) -> bool:  # noqa: N802
        try:
            self._service.add_beat(t, label, confidence)
            self._refresh_beats()
            self._has_changes = True
            self._set_status("edited", error="")
            self.undoRedoChanged.emit()
            return True
        except Exception as exc:  # noqa: BLE001
            self._set_status("error", error=str(exc))
            return False

    @Slot(result=bool)
    def rebuildPipeline(self) -> bool:  # noqa: N802
        try:
            self._rebuilding = True
            self._set_status("rebuilding", error="")
            # Compute a blanket affected set for all changed artifacts
            affected = AffectedArtifacts(
                timeline=True,
                scene_srt=True,
                beat_json=True,
                stage1_segments=True,
                ltx_prompt=True,
                render_plan=True,
            )
            self._service.rebuild_pipeline(affected)
            self._rebuilding = False
            self._set_status("rebuild_queued", error="")
            return True
        except Exception as exc:  # noqa: BLE001
            self._rebuilding = False
            self._set_status("error", error=str(exc))
            return False

    @Slot(result=bool)
    def undo(self) -> bool:
        try:
            self._service.undo()
            self._refresh_all()
            self._set_status("undone", error="")
            self.undoRedoChanged.emit()
            return True
        except Exception as exc:  # noqa: BLE001
            self._set_status("error", error=str(exc))
            return False

    @Slot(result=bool)
    def redo(self) -> bool:
        try:
            self._service.redo()
            self._refresh_all()
            self._set_status("redone", error="")
            self.undoRedoChanged.emit()
            return True
        except Exception as exc:  # noqa: BLE001
            self._set_status("error", error=str(exc))
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_status(self, status: str, *, error: str = "") -> None:
        self._status = status
        self._error = error
        self.statusChanged.emit()
        if error:
            self.errorChanged.emit()

    def _refresh_segments(self) -> None:
        try:
            snapshot = self._service.current_snapshot()
            self._segments = [
                {
                    "start": s.start,
                    "end": s.end,
                    "kind": s.kind,
                    "text": s.text,
                    "lyrics_line": s.lyrics_line or "",
                    "notes": s.notes or "",
                    "is_draft": s.is_draft,
                }
                for s in snapshot.segments
            ]
            self.segmentsChanged.emit()
        except Exception as exc:  # noqa: BLE001
            self._set_status("error", error=str(exc))

    def _refresh_boundaries(self) -> None:
        try:
            snapshot = self._service.current_snapshot()
            self._boundaries = [
                {
                    "start": b.start,
                    "end": b.end,
                    "reason": b.reason,
                }
                for b in snapshot.scene_boundaries
            ]
            self.boundariesChanged.emit()
        except Exception as exc:  # noqa: BLE001
            self._set_status("error", error=str(exc))

    def _refresh_beats(self) -> None:
        try:
            snapshot = self._service.current_snapshot()
            self._beats = [
                {
                    "time_s": m.time_s,
                    "label": m.label,
                    "confidence": m.confidence,
                }
                for m in snapshot.beat_markers
            ]
            self.beatsChanged.emit()
        except Exception as exc:  # noqa: BLE001
            self._set_status("error", error=str(exc))

    def _refresh_all(self) -> None:
        self._refresh_segments()
        self._refresh_boundaries()
        self._refresh_beats()
