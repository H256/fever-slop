"""Studio-level service for timeline editing.

Sits between the desktop viewmodel and the application/infrastructure layers.
Manages in-memory edit history (undo/redo) and orchestrates edits through
timeline document ports and the job registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from feverslop.adapters.project_timeline_documents import ProjectTimelineDocuments
from feverslop.domain.timeline_editing import (
    BeatMarker,
    EditableTimelineSegment,
    SceneBoundary,
    TimelineSnapshot,
    compute_edit_impact,
    merge_segments,
    split_segment,
    validate_beat_markers,
    validate_scene_boundaries,
)
from feverslop.ports.timeline_documents import (
    AffectedArtifacts,
    TimelineReadPort,
    TimelineWritePort,
)
from feverslop.studio.jobs import JobRegistry


class TimelineStudioService:
    """In-process timeline editor with undo/redo support.

    Lifecycle:
        1. set_project_dir() — resolve the project directory for read/write ports
        2. load() — reads current documents via read_port, sets _current
        3. Edit operations — push snapshot to undo stack, mutate _current, write
        4. undo()/redo() — traverse history stacks
        5. rebuild_pipeline() — schedule async rebuild via job registry
    """

    _MAX_HISTORY = 50

    def __init__(
        self,
        job_registry: JobRegistry,
        *,
        read_port: TimelineReadPort | None = None,
        write_port: TimelineWritePort | None = None,
    ) -> None:
        self._job_registry = job_registry
        self._read_port = read_port
        self._write_port = write_port
        self._project_dir: str = ""
        self._current: TimelineSnapshot | None = None
        self._undo_stack: list[TimelineSnapshot] = []
        self._redo_stack: list[TimelineSnapshot] = []

    def set_project_dir(self, project_dir: str) -> None:
        """Bind this service to a specific project directory (creates adapters)."""
        self._project_dir = str(Path(project_dir).resolve())
        adapter = ProjectTimelineDocuments(Path(self._project_dir))
        self._read_port = adapter
        self._write_port = adapter
        # Reset state on project switch
        self._current = None
        self._undo_stack.clear()
        self._redo_stack.clear()

    def _check_project(self) -> None:
        if self._read_port is None or self._write_port is None:
            raise RuntimeError("No project set; provide read_port/write_port or call set_project_dir()")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def current_snapshot(self) -> TimelineSnapshot:
        """Return the current in-memory snapshot."""
        if self._current is None:
            raise RuntimeError("No timeline loaded; call load() first")
        return self._current

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def load(self) -> TimelineSnapshot:
        """Read current documents into memory and reset redo stack."""
        self._check_project()
        raw = self._read_port.read_timeline()  # type: ignore[union-attr]
        if raw:
            snapshot = TimelineSnapshot.from_json(raw[0])
        else:
            snapshot = TimelineSnapshot(
                segments=[],
                scene_boundaries=[],
                beat_markers=[],
                metadata={"project_dir": self._project_dir},
            )
        self._current = snapshot
        self._redo_stack.clear()
        return snapshot

    def save(self) -> None:
        """Write current snapshot to disk via write_port."""
        self._check_project()
        if self._current is None:
            raise RuntimeError("No timeline loaded; call load() first")
        self._write_port.write_timeline([self._current.to_json()])  # type: ignore[union-attr]
        srt = self._read_port.read_scene_srt()  # type: ignore[union-attr]
        if srt is not None:
            self._write_port.write_scene_srt(srt)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> TimelineSnapshot:
        self._check_project()
        if self._current is None:
            raise RuntimeError("No timeline loaded; call load() first")
        return self._current

    def _push_history(self) -> None:
        """Push current snapshot onto undo stack and clear redo."""
        if self._current is not None:
            self._undo_stack.append(self._current)
            if len(self._undo_stack) > self._MAX_HISTORY:
                self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _apply_snapshot(self, snapshot: TimelineSnapshot) -> None:
        """Replace current snapshot and persist."""
        self._current = snapshot
        self._write_port.write_timeline(  # type: ignore[union-attr]
            [self._current.to_json()],
        )

    def _compute_impact(
        self, before: TimelineSnapshot, after: TimelineSnapshot,
    ) -> AffectedArtifacts:
        impact = compute_edit_impact(before, after)
        return AffectedArtifacts.from_timeline_edit_impact(impact)

    # ------------------------------------------------------------------
    # Edit operations
    # ------------------------------------------------------------------

    def edit_segment(
        self,
        index: int,
        start_delta: float = 0.0,
        end_delta: float = 0.0,
        lyrics: str | None = None,
        notes: str | None = None,
    ) -> AffectedArtifacts:
        current = self._ensure_loaded()
        segments = list(current.segments)
        if index < 0 or index >= len(segments):
            raise ValueError(
                f"segment index {index} out of range [0, {len(segments)})",
            )

        original = segments[index]
        new_start = original.start + start_delta
        new_end = original.end + end_delta

        self._push_history()
        before = self._current

        new_seg = EditableTimelineSegment(
            start=new_start,
            end=new_end,
            kind=original.kind,
            text=original.text,
            lyrics_line=(lyrics if lyrics is not None else original.lyrics_line),
            notes=(notes if notes is not None else original.notes),
            is_draft=original.is_draft,
        )
        segments[index] = new_seg

        after = TimelineSnapshot(
            segments=segments,
            scene_boundaries=current.scene_boundaries,
            beat_markers=current.beat_markers,
            metadata=dict(current.metadata),
        )
        self._apply_snapshot(after)
        return self._compute_impact(before, after)  # type: ignore[arg-type]

    def split_segment(self, index: int, at: float) -> AffectedArtifacts:
        current = self._ensure_loaded()
        if index < 0 or index >= len(current.segments):
            raise ValueError(f"segment index {index} out of range")

        self._push_history()
        before = self._current

        segment = current.segments[index]
        left, right = split_segment(segment, at)

        segments = list(current.segments)
        segments[index] = left
        segments.insert(index + 1, right)

        after = TimelineSnapshot(
            segments=segments,
            scene_boundaries=current.scene_boundaries,
            beat_markers=current.beat_markers,
            metadata=dict(current.metadata),
        )
        self._apply_snapshot(after)
        return self._compute_impact(before, after)  # type: ignore[arg-type]

    def merge_segments(self, index: int, count: int) -> AffectedArtifacts:
        current = self._ensure_loaded()
        if count < 2:
            raise ValueError(f"merge requires count >= 2, got {count}")
        if index < 0 or index + count > len(current.segments):
            raise ValueError(
                f"segments [{index}:{index + count}] out of range "
                f"[0, {len(current.segments)})",
            )

        self._push_history()
        before = self._current

        segments = list(current.segments)
        to_merge = segments[index:index + count]
        merged = merge_segments(to_merge)

        new_segments = segments[:index] + [merged] + segments[index + count:]

        after = TimelineSnapshot(
            segments=new_segments,
            scene_boundaries=current.scene_boundaries,
            beat_markers=current.beat_markers,
            metadata=dict(current.metadata),
        )
        self._apply_snapshot(after)
        return self._compute_impact(before, after)  # type: ignore[arg-type]

    def add_scene_boundary(
        self, start: float, end: float, reason: str,
    ) -> AffectedArtifacts:
        current = self._ensure_loaded()
        self._push_history()
        before = self._current

        boundary = SceneBoundary(start=start, end=end, reason=reason)
        new_boundaries = list(current.scene_boundaries) + [boundary]
        validated = validate_scene_boundaries(new_boundaries)

        after = TimelineSnapshot(
            segments=current.segments,
            scene_boundaries=validated,
            beat_markers=current.beat_markers,
            metadata=dict(current.metadata),
        )
        self._apply_snapshot(after)
        return self._compute_impact(before, after)  # type: ignore[arg-type]

    def add_beat(self, time: float, label: str, confidence: float) -> AffectedArtifacts:
        current = self._ensure_loaded()
        self._push_history()
        before = self._current

        marker = BeatMarker(time_s=time, label=label, confidence=confidence)
        new_markers = list(current.beat_markers) + [marker]
        validated = validate_beat_markers(new_markers)

        after = TimelineSnapshot(
            segments=current.segments,
            scene_boundaries=current.scene_boundaries,
            beat_markers=validated,
            metadata=dict(current.metadata),
        )
        self._apply_snapshot(after)
        return self._compute_impact(before, after)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Pipeline rebuild
    # ------------------------------------------------------------------

    def rebuild_pipeline(self, affected: AffectedArtifacts) -> dict[str, Any]:
        """Schedule a rebuild job for invalidated downstream artifacts."""
        self._ensure_loaded()
        job = self._job_registry.add_rebuild_plan_timeline(
            self._project_dir, affected,
        )
        return job

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def undo(self) -> None:
        """Revert to the most recent snapshot before the last edit."""
        if not self._undo_stack:
            raise RuntimeError("Nothing to undo")
        self._redo_stack.append(self._ensure_loaded())
        self._current = self._undo_stack.pop()
        self._write_port.write_timeline(  # type: ignore[union-attr]
            [self._current.to_json()],
        )

    def redo(self) -> None:
        """Re-apply the most recently undone edit."""
        if not self._redo_stack:
            raise RuntimeError("Nothing to redo")
        self._undo_stack.append(self._ensure_loaded())
        self._current = self._redo_stack.pop()
        self._write_port.write_timeline(  # type: ignore[union-attr]
            [self._current.to_json()],
        )
