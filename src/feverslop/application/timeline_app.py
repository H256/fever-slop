"""Application service for timeline editing.

Sits between ports (infrastructure adapters) and domain logic. Orchestrates
read-edit-write cycles and computes downstream artifact impact.
"""

from __future__ import annotations

from feverslop.domain.timeline_editing import (
    AffectedArtifacts,
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
    TimelineReadPort,
    TimelineWritePort,
)


class TimelineAppService:
    """Application-level orchestrator for timeline edits.

    Bridges port I/O with domain editing primitives. Each public method
    follows the Load → Transform → Save pattern and returns
    :class:`~feverslop.domain.timeline_editing.AffectedArtifacts`
    describing which downstream pipeline outputs are stale.
    """

    def __init__(
        self,
        read_port: TimelineReadPort,
        write_port: TimelineWritePort,
    ) -> None:
        self._read = read_port
        self._write = write_port

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> TimelineSnapshot:
        """Read the current timeline from the read port."""
        raw = self._read.read_timeline()
        if raw:
            return TimelineSnapshot.from_json(raw[0])
        return TimelineSnapshot(
            segments=[],
            scene_boundaries=[],
            beat_markers=[],
            metadata={},
        )

    def _save(self, snapshot: TimelineSnapshot) -> None:
        """Persist snapshot back through the write port."""
        self._write.write_timeline([snapshot.to_json()])

    @staticmethod
    def _impact(before: TimelineSnapshot, after: TimelineSnapshot) -> AffectedArtifacts:
        edit_impact = compute_edit_impact(before, after)
        return AffectedArtifacts.from_timeline_edit_impact(edit_impact)

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
        """Adjust one segment by timedeltas and optional text overrides."""
        current = self._load()
        segments = list(current.segments)
        if index < 0 or index >= len(segments):
            raise ValueError(
                f"segment index {index} out of range [0, {len(segments)})",
            )

        original = segments[index]
        before = current
        new_seg = EditableTimelineSegment(
            start=original.start + start_delta,
            end=original.end + end_delta,
            kind=original.kind,
            text=original.text,
            lyrics_line=lyrics if lyrics is not None else original.lyrics_line,
            notes=notes if notes is not None else original.notes,
            is_draft=original.is_draft,
        )
        segments[index] = new_seg

        after = TimelineSnapshot(
            segments=segments,
            scene_boundaries=current.scene_boundaries,
            beat_markers=current.beat_markers,
            metadata=dict(current.metadata),
        )
        self._save(after)
        return self._impact(before, after)

    def split_segment_at(self, index: int, at: float) -> AffectedArtifacts:
        """Split segment at *index* into two halves at time *at*."""
        current = self._load()
        if index < 0 or index >= len(current.segments):
            raise ValueError(
                f"segment index {index} out of range [0, {len(current.segments)})",
            )

        before = current
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
        self._save(after)
        return self._impact(before, after)

    def merge_segments_at(self, index: int, count: int) -> AffectedArtifacts:
        """Merge *count* adjacent segments starting at *index*."""
        current = self._load()
        if count < 2:
            raise ValueError(f"merge requires count >= 2, got {count}")
        if index < 0 or index + count > len(current.segments):
            raise ValueError(
                f"segments [{index}:{index + count}] out of range "
                f"[0, {len(current.segments)})",
            )

        before = current
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
        self._save(after)
        return self._impact(before, after)

    def add_scene_boundary(
        self, start: float, end: float, reason: str,
    ) -> AffectedArtifacts:
        """Add a scene boundary (validated against existing boundaries)."""
        current = self._load()
        before = current

        boundary = SceneBoundary(start=start, end=end, reason=reason)
        new_boundaries = list(current.scene_boundaries) + [boundary]
        validated = validate_scene_boundaries(new_boundaries)

        after = TimelineSnapshot(
            segments=current.segments,
            scene_boundaries=validated,
            beat_markers=current.beat_markers,
            metadata=dict(current.metadata),
        )
        self._save(after)
        return self._impact(before, after)

    def add_beat(self, time_s: float, label: str, confidence: float) -> AffectedArtifacts:
        """Add a beat marker (validated against existing markers)."""
        current = self._load()
        before = current

        marker = BeatMarker(time_s=time_s, label=label, confidence=confidence)
        new_markers = list(current.beat_markers) + [marker]
        validated = validate_beat_markers(new_markers)

        after = TimelineSnapshot(
            segments=current.segments,
            scene_boundaries=current.scene_boundaries,
            beat_markers=validated,
            metadata=dict(current.metadata),
        )
        self._save(after)
        return self._impact(before, after)

    def compute_pipeline_rebuild(self) -> AffectedArtifacts:
        """Return affected artifacts by comparing loaded snapshot against file.

        This is a no-op snapshot comparison for the app layer; in practice
        the studio service tracks before/after snapshots. Returns an affected
        set with the timeline flag set to indicate the document has been
        inspected.
        """
        snapshot = self._load()
        # Re-read from the underlying port to detect file vs snapshot drift.
        raw = self._read.read_timeline()
        if raw:
            file_snapshot = TimelineSnapshot.from_json(raw[0])
        else:
            file_snapshot = TimelineSnapshot(
                segments=[],
                scene_boundaries=[],
                beat_markers=[],
                metadata={},
            )
        return self._impact(snapshot, file_snapshot)
