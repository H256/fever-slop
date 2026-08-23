"""Tests for TimelineStudioService (headless — no PySide6)."""

from __future__ import annotations

import copy
import unittest

from feverslop.domain.timeline_editing import (
    BeatMarker,
    EditableTimelineSegment,
    SceneBoundary,
    TimelineSnapshot,
)
from feverslop.ports.timeline_documents import AffectedArtifacts
from feverslop.composition.job_runtime import JobRegistry
from feverslop.composition.timeline_service import TimelineStudioService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(start: float, end: float, **kw) -> EditableTimelineSegment:
    return EditableTimelineSegment(
        start=start, end=end, kind=kw.pop("kind", "vocals"),
        text=kw.pop("text", "test"), lyrics_line=kw.pop("lyrics_line", None),
        notes=kw.pop("notes", None), is_draft=kw.pop("is_draft", False), **kw,
    )

def _bnd(start: float, end: float, reason: str = "test") -> SceneBoundary:
    return SceneBoundary(start=start, end=end, reason=reason)

def _mk(time_s: float, label: str = "beat", confidence: float = 0.9) -> BeatMarker:
    return BeatMarker(time_s=time_s, label=label, confidence=confidence)


class MockReadPort:
    """A read port backed by an in-memory dictionary."""

    def __init__(self) -> None:
        self._timeline: list[dict] | None = None
        self._scene_srt: str | None = None
        self._beat_json: list[dict] | None = None
        self._stage1_segments: list[dict] | None = None
        self._ltx_prompt: list[dict] | None = None
        self._render_plan: dict | None = None

    def set_timeline(self, snapshot: TimelineSnapshot) -> None:
        self._timeline = [snapshot.to_json()]

    def read_timeline(self) -> list[dict]:
        if self._timeline is None:
            return []
        return copy.deepcopy(self._timeline)

    def read_scene_srt(self) -> str | None:
        return self._scene_srt

    def read_beat_json(self) -> list[dict] | None:
        return copy.deepcopy(self._beat_json) if self._beat_json is not None else None

    def read_stage1_segments(self) -> list[dict] | None:
        return copy.deepcopy(self._stage1_segments) if self._stage1_segments is not None else None

    def read_ltx_prompt_relay(self) -> list[dict] | None:
        return copy.deepcopy(self._ltx_prompt) if self._ltx_prompt is not None else None

    def read_render_plan(self) -> dict | None:
        return copy.deepcopy(self._render_plan) if self._render_plan is not None else None


class MockWritePort:
    """A write port that records what was written."""

    def __init__(self) -> None:
        self.written_timeline: list[dict] | None = None
        self.written_scene_srt: str | None = None

    def write_timeline(self, data: list[dict]) -> None:
        self.written_timeline = copy.deepcopy(data)

    def write_scene_srt(self, content: str) -> None:
        self.written_scene_srt = content


def _make_service(
    snapshot: TimelineSnapshot | None = None,
    job_registry: JobRegistry | None = None,
) -> tuple[TimelineStudioService, MockReadPort, MockWritePort]:
    read = MockReadPort()
    write = MockWritePort()
    registry = job_registry or JobRegistry()
    if snapshot is not None:
        read.set_timeline(snapshot)
    service = TimelineStudioService(
        registry,
        read_port=read,
        write_port=write,
    )
    return service, read, write


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------

class LoadSaveTest(unittest.TestCase):
    def test_load_returns_snapshot(self):
        snap = TimelineSnapshot(
            segments=[_seg(0, 5)],
            scene_boundaries=[_bnd(0, 5)],
            beat_markers=[_mk(1.0)],
            metadata={"version": 1},
        )
        service, *_ = _make_service(snap)
        result = service.load()
        self.assertEqual(len(result.segments), 1)
        self.assertEqual(len(result.scene_boundaries), 1)
        self.assertEqual(len(result.beat_markers), 1)

    def test_load_empty_timeline_returns_empty_snapshot(self):
        service, read, _ = _make_service()
        result = service.load()
        self.assertEqual(len(result.segments), 0)
        self.assertEqual(len(result.scene_boundaries), 0)
        self.assertEqual(len(result.beat_markers), 0)

    def test_save_writes_timeline(self):
        snap = TimelineSnapshot(
            segments=[_seg(0, 5)], scene_boundaries=[], beat_markers=[], metadata={},
        )
        service, read, write = _make_service(snap)
        service.load()
        service.save()
        self.assertIsNotNone(write.written_timeline)
        self.assertEqual(len(write.written_timeline), 1)

    def test_save_without_load_raises(self):
        service, _, _ = _make_service()
        with self.assertRaises(RuntimeError):
            service.save()


# ---------------------------------------------------------------------------
# Edit Segment
# ---------------------------------------------------------------------------

class EditSegmentTest(unittest.TestCase):
    def test_edit_segment_returns_impact(self):
        snap = TimelineSnapshot(
            segments=[_seg(0, 10, text="hello")],
            scene_boundaries=[],
            beat_markers=[],
            metadata={},
        )
        service, _, _ = _make_service(snap)
        service.load()
        impact = service.edit_segment(0, start_delta=1.0, end_delta=-1.0)
        self.assertIsInstance(impact, AffectedArtifacts)
        self.assertTrue(impact.timeline)

    def test_edit_segment_updates_current_snapshot(self):
        snap = TimelineSnapshot(
            segments=[_seg(0, 10, text="hello")],
            scene_boundaries=[],
            beat_markers=[],
            metadata={},
        )
        service, _, _ = _make_service(snap)
        service.load()
        service.edit_segment(0, start_delta=2.0, end_delta=0.0)
        self.assertEqual(service.current_snapshot().segments[0].start, 2.0)

    def test_edit_segment_lyrics(self):
        snap = TimelineSnapshot(
            segments=[_seg(0, 10, text="hi")],
            scene_boundaries=[],
            beat_markers=[],
            metadata={},
        )
        service, _, _ = _make_service(snap)
        service.load()
        service.edit_segment(0, lyrics="new lyrics")
        self.assertEqual(service.current_snapshot().segments[0].lyrics_line, "new lyrics")

    def test_edit_segment_notes(self):
        snap = TimelineSnapshot(
            segments=[_seg(0, 10)], scene_boundaries=[], beat_markers=[], metadata={},
        )
        service, _, _ = _make_service(snap)
        service.load()
        service.edit_segment(0, notes="fix timing")
        self.assertEqual(service.current_snapshot().segments[0].notes, "fix timing")

    def test_edit_segment_out_of_range_raises(self):
        snap = TimelineSnapshot(
            segments=[_seg(0, 10)], scene_boundaries=[], beat_markers=[], metadata={},
        )
        service, _, _ = _make_service(snap)
        service.load()
        with self.assertRaises(ValueError):
            service.edit_segment(5, start_delta=1.0)

    def test_edit_segment_invalidates_on_text_change(self):
        snap = TimelineSnapshot(
            segments=[_seg(0, 10, text="hello")],
            scene_boundaries=[], beat_markers=[], metadata={},
        )
        service, _, write = _make_service(snap)
        service.load()
        # Changing start/end also changes timeline content
        impact = service.edit_segment(0, start_delta=1.0)
        self.assertTrue(impact.timeline)

    def test_edit_segment_persists_via_write_port(self):
        snap = TimelineSnapshot(
            segments=[_seg(0, 10)], scene_boundaries=[], beat_markers=[], metadata={},
        )
        service, _, write = _make_service(snap)
        service.load()
        service.edit_segment(0, start_delta=1.0)
        self.assertIsNotNone(write.written_timeline)


# ---------------------------------------------------------------------------
# Split Segment
# ---------------------------------------------------------------------------

class SplitSegmentTest(unittest.TestCase):
    def test_split_segment_doubles_count(self):
        snap = TimelineSnapshot(
            segments=[_seg(0, 10)], scene_boundaries=[], beat_markers=[], metadata={},
        )
        service, _, _ = _make_service(snap)
        service.load()
        impact = service.split_segment(0, at=5.0)
        self.assertEqual(len(service.current_snapshot().segments), 2)
        self.assertTrue(impact.timeline)

    def test_split_preserves_total_duration(self):
        snap = TimelineSnapshot(
            segments=[_seg(0, 10)], scene_boundaries=[], beat_markers=[], metadata={},
        )
        service, _, _ = _make_service(snap)
        service.load()
        service.split_segment(0, at=3.0)
        segs = service.current_snapshot().segments
        self.assertAlmostEqual(segs[0].end - segs[0].start + segs[1].end - segs[1].start, 10.0)

    def test_split_outside_raises(self):
        snap = TimelineSnapshot(
            segments=[_seg(0, 10)], scene_boundaries=[], beat_markers=[], metadata={},
        )
        service, _, _ = _make_service(snap)
        service.load()
        with self.assertRaises(ValueError):
            service.split_segment(0, at=10.0)

    def test_split_negative_index_raises(self):
        snap = TimelineSnapshot(
            segments=[_seg(0, 10)], scene_boundaries=[], beat_markers=[], metadata={},
        )
        service, _, _ = _make_service(snap)
        service.load()
        with self.assertRaises(ValueError):
            service.split_segment(-1, at=5.0)


# ---------------------------------------------------------------------------
# Merge Segments
# ---------------------------------------------------------------------------

class MergeSegmentsTest(unittest.TestCase):
    def test_merge_two_adjacent(self):
        snap = TimelineSnapshot(
            segments=[_seg(0, 5, text="a"), _seg(5, 10, text="b")],
            scene_boundaries=[], beat_markers=[], metadata={},
        )
        service, _, _ = _make_service(snap)
        service.load()
        impact = service.merge_segments(0, count=2)
        self.assertEqual(len(service.current_snapshot().segments), 1)
        self.assertEqual(service.current_snapshot().segments[0].text, "a b")
        self.assertTrue(impact.timeline)

    def test_merge_insufficient_raises(self):
        snap = TimelineSnapshot(
            segments=[_seg(0, 5)], scene_boundaries=[], beat_markers=[], metadata={},
        )
        service, _, _ = _make_service(snap)
        service.load()
        with self.assertRaises(ValueError):
            service.merge_segments(0, count=2)

    def test_merge_non_adjacent_raises(self):
        snap = TimelineSnapshot(
            segments=[_seg(0, 3), _seg(5, 10)],
            scene_boundaries=[], beat_markers=[], metadata={},
        )
        service, _, _ = _make_service(snap)
        service.load()
        with self.assertRaises(ValueError):
            service.merge_segments(0, count=2)


# ---------------------------------------------------------------------------
# Add Scene Boundary
# ---------------------------------------------------------------------------

class AddSceneBoundaryTest(unittest.TestCase):
    def test_add_boundary(self):
        snap = TimelineSnapshot(
            segments=[_seg(0, 10)], scene_boundaries=[], beat_markers=[], metadata={},
        )
        service, _, _ = _make_service(snap)
        service.load()
        impact = service.add_scene_boundary(0, 5, "verse starts")
        self.assertEqual(len(service.current_snapshot().scene_boundaries), 1)
        self.assertTrue(impact.scene_srt)

    def test_add_overlapping_boundary_raises(self):
        snap = TimelineSnapshot(
            segments=[],
            scene_boundaries=[SceneBoundary(start=0, end=10, reason="first")],
            beat_markers=[],
            metadata={},
        )
        service, _, _ = _make_service(snap)
        service.load()
        with self.assertRaises(ValueError):
            service.add_scene_boundary(5, 12, "overlap")

    def test_add_too_short_boundary_raises(self):
        snap = TimelineSnapshot(
            segments=[], scene_boundaries=[], beat_markers=[], metadata={},
        )
        service, _, _ = _make_service(snap)
        service.load()
        with self.assertRaises(ValueError):
            service.add_scene_boundary(0, 1, "too short")


# ---------------------------------------------------------------------------
# Add Beat
# ---------------------------------------------------------------------------

class AddBeatTest(unittest.TestCase):
    def test_add_beat(self):
        snap = TimelineSnapshot(
            segments=[], scene_boundaries=[], beat_markers=[], metadata={},
        )
        service, _, _ = _make_service(snap)
        service.load()
        impact = service.add_beat(1.5, "kick", 0.95)
        self.assertEqual(len(service.current_snapshot().beat_markers), 1)
        self.assertTrue(impact.beat_json)

    def test_add_duplicate_beat_raises(self):
        snap = TimelineSnapshot(
            segments=[], scene_boundaries=[], beat_markers=[_mk(1.0)], metadata={},
        )
        service, _, _ = _make_service(snap)
        service.load()
        with self.assertRaises(ValueError):
            service.add_beat(1.0, "dup", 0.9)

    def test_add_invalid_confidence_raises(self):
        snap = TimelineSnapshot(
            segments=[], scene_boundaries=[], beat_markers=[], metadata={},
        )
        service, _, _ = _make_service(snap)
        service.load()
        with self.assertRaises(ValueError):
            service.add_beat(1.0, "bad", 1.5)


# ---------------------------------------------------------------------------
# Rebuild Pipeline
# ---------------------------------------------------------------------------

class RebuildPipelineTest(unittest.TestCase):
    def test_rebuild_schedules_job(self):
        snap = TimelineSnapshot(
            segments=[_seg(0, 5)], scene_boundaries=[], beat_markers=[], metadata={},
        )
        registry = JobRegistry()
        service, *_ = _make_service(snap, job_registry=registry)
        service.load()
        affected = service.edit_segment(0, start_delta=1.0)
        # Should not raise — job is scheduled in background thread
        service.rebuild_pipeline(affected)
        # Wait very briefly for thread to start
        import time
        time.sleep(0.05)
        jobs = registry.list()
        # There should be at least one job now
        rebuild_jobs = [j for j in jobs if j.get("action") == "rebuild-plan-timeline"]
        self.assertTrue(len(rebuild_jobs) >= 1)


# ---------------------------------------------------------------------------
# Undo / Redo History
# ---------------------------------------------------------------------------

class UndoRedoTest(unittest.TestCase):
    def test_can_undo_starts_false(self):
        snap = TimelineSnapshot(
            segments=[_seg(0, 5)], scene_boundaries=[], beat_markers=[], metadata={},
        )
        service, *_ = _make_service(snap)
        service.load()
        self.assertFalse(service.can_undo())

    def test_can_undo_becomes_true_after_edit(self):
        snap = TimelineSnapshot(
            segments=[_seg(0, 5)], scene_boundaries=[], beat_markers=[], metadata={},
        )
        service, *_ = _make_service(snap)
        service.load()
        service.edit_segment(0, start_delta=1.0)
        self.assertTrue(service.can_undo())

    def test_undo_reverts_snapshot(self):
        snap = TimelineSnapshot(
            segments=[_seg(0, 5)], scene_boundaries=[], beat_markers=[], metadata={},
        )
        service, read, _ = _make_service(snap)
        service.load()
        service.edit_segment(0, start_delta=5.0)
        self.assertEqual(service.current_snapshot().segments[0].start, 5.0)
        service.undo()
        self.assertEqual(service.current_snapshot().segments[0].start, 0.0)

    def test_undo_clears_redo_stack_on_new_edit(self):
        snap = TimelineSnapshot(
            segments=[_seg(0, 5)], scene_boundaries=[], beat_markers=[], metadata={},
        )
        service, read, _ = _make_service(snap)
        service.load()
        service.edit_segment(0, start_delta=5.0)
        service.undo()
        self.assertTrue(service.can_redo())
        # New edit should clear redo
        service.edit_segment(0, start_delta=1.0)
        self.assertFalse(service.can_redo())

    def test_redo_reapplies(self):
        snap = TimelineSnapshot(
            segments=[_seg(0, 5)], scene_boundaries=[], beat_markers=[], metadata={},
        )
        service, read, _ = _make_service(snap)
        service.load()
        service.edit_segment(0, start_delta=5.0)
        service.undo()
        self.assertFalse(service.can_undo())
        self.assertTrue(service.can_redo())
        service.redo()
        self.assertEqual(service.current_snapshot().segments[0].start, 5.0)
        self.assertTrue(service.can_undo())

    def test_undo_exhausted(self):
        snap = TimelineSnapshot(
            segments=[_seg(0, 5)], scene_boundaries=[], beat_markers=[], metadata={},
        )
        service, *_ = _make_service(snap)
        service.load()
        # No edits → cannot undo
        with self.assertRaises(RuntimeError):
            service.undo()

    def test_redo_exhausted(self):
        snap = TimelineSnapshot(
            segments=[_seg(0, 5)], scene_boundaries=[], beat_markers=[], metadata={},
        )
        service, *_ = _make_service(snap)
        service.load()
        with self.assertRaises(RuntimeError):
            service.redo()

    def test_history_bounded_at_50(self):
        # Use a very wide segment so repeated deltas don't exceed end.
        snap = TimelineSnapshot(
            segments=[_seg(0, 1000)], scene_boundaries=[], beat_markers=[], metadata={},
        )
        service, read, _ = _make_service(snap)
        service.load()
        for i in range(60):
            # Alternate direction so segment stays valid.
            delta = 1.0 if i % 2 == 0 else -1.0
            service.edit_segment(0, start_delta=delta)
        # Undo stack should be capped at 50
        self.assertLessEqual(len(service._undo_stack), 50)
        # Should still be able to undo
        self.assertTrue(service.can_undo())

    def test_load_clears_redo(self):
        snap = TimelineSnapshot(
            segments=[_seg(0, 5)], scene_boundaries=[], beat_markers=[], metadata={},
        )
        service, read, _ = _make_service(snap)
        service.load()
        service.edit_segment(0, start_delta=5.0)
        service.undo()
        self.assertTrue(service.can_redo())
        # Reload should clear redo
        service.load()
        self.assertFalse(service.can_redo())


if __name__ == "__main__":
    unittest.main()
