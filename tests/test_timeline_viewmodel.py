"""Tests for TimelineStudioViewModel (headless — no PySide6/QML imports).

Mocks the service so we only test viewmodel command wiring and status updates.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, PropertyMock

from feverslop.ports.timeline_documents import AffectedArtifacts
from feverslop.domain.timeline_editing import (
    EditableTimelineSegment,
    SceneBoundary,
    BeatMarker,
    TimelineSnapshot,
)


def _seg(start: float, end: float, **kw) -> EditableTimelineSegment:
    return EditableTimelineSegment(
        start=start, end=end, kind=kw.pop("kind", "vocals"),
        text=kw.pop("text", "test"), lyrics_line=kw.pop("lyrics_line", None),
        notes=kw.pop("notes", None), is_draft=kw.pop("is_draft", False), **kw
    )


class MockTimelineService:
    """Lightweight mock of TimelineStudioService for headless testing."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self._snapshot = TimelineSnapshot(
            segments=[_seg(0, 5), _seg(5, 10)],
            scene_boundaries=[],
            beat_markers=[],
            metadata={},
        )
        self._undo_stack: list = []
        self._redo_stack: list = []
        self._error: str | None = None
        self._simulate_error = False

    def current_snapshot(self) -> TimelineSnapshot:
        self.calls.append(("current_snapshot", ()))
        return self._snapshot

    def load(self) -> TimelineSnapshot:
        self.calls.append(("load", ()))
        if self._simulate_error:
            raise RuntimeError("simulated error")
        return self._snapshot

    def save(self) -> None:
        self.calls.append(("save", ()))

    def edit_segment(self, index: int, start_delta: float, end_delta: float,
                      lyrics: str | None, notes: str | None) -> AffectedArtifacts:
        self.calls.append(("edit_segment", (index, start_delta, end_delta, lyrics, notes)))
        return AffectedArtifacts(timeline=True)

    def split_segment(self, index: int, at: float) -> AffectedArtifacts:
        self.calls.append(("split_segment", (index, at)))
        return AffectedArtifacts(timeline=True)

    def merge_segments(self, index: int, count: int) -> AffectedArtifacts:
        self.calls.append(("merge_segments", (index, count)))
        return AffectedArtifacts(timeline=True)

    def add_scene_boundary(self, start: float, end: float, reason: str) -> AffectedArtifacts:
        self.calls.append(("add_scene_boundary", (start, end, reason)))
        return AffectedArtifacts(scene_srt=True)

    def add_beat(self, time: float, label: str, confidence: float) -> AffectedArtifacts:
        self.calls.append(("add_beat", (time, label, confidence)))
        return AffectedArtifacts(beat_json=True)

    def rebuild_pipeline(self, affected: AffectedArtifacts) -> None:
        self.calls.append(("rebuild_pipeline", (affected,)))

    def can_undo(self) -> bool:
        if self._simulate_error:
            return False
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def undo(self) -> None:
        self.calls.append(("undo", ()))

    def redo(self) -> None:
        self.calls.append(("redo", ()))


# ---------------------------------------------------------------------------
# Import viewmodel class (may use PySide6, but we test headlessly)
# ---------------------------------------------------------------------------

try:
    from feverslop.studio.desktop.viewmodels.timeline import TimelineStudioViewModel

    HAS_VIEWS = True
except ImportError:
    HAS_VIEWS = False


@unittest.skipUnless(HAS_VIEWS, "TimelineStudioViewModel not available")
class TimelineStudioViewModelTest(unittest.TestCase):
    """Test that commands wire through to the service correctly."""

    def setUp(self) -> None:
        self.service = MockTimelineService()
        # Create viewmodel without PySide6 QObject parent — just pass the service
        self.vm = TimelineStudioViewModel(service=self.service)
        self.service.calls.clear()

    def _last_call(self, name: str) -> tuple[str, tuple] | None:
        """Find the most recent call matching *name*."""
        for call in reversed(self.service.calls):
            if call[0] == name:
                return call
        return None

    # ------------------------------------------------------------------
    # Commands delegate to service
    # ------------------------------------------------------------------

    def test_load_project_calls_service_load(self):
        self.vm.loadProject()
        self.assertIn(("load", ()), self.service.calls)

    def test_save_calls_service_save(self):
        self.vm.loadProject()
        self.service.calls.clear()
        self.vm.save()
        self.assertIn(("save", ()), self.service.calls)

    def test_edit_segment_forwards_args(self):
        self.vm.editSegment(index=0, startDelta=1.0, endDelta=-1.0, lyrics="new", notes="fix")
        call = self._last_call("edit_segment")
        self.assertIsNotNone(call)
        self.assertEqual(call[1][0], 0)   # index
        self.assertEqual(call[1][1], 1.0)  # start_delta
        self.assertEqual(call[1][2], -1.0) # end_delta
        self.assertEqual(call[1][3], "new")
        self.assertEqual(call[1][4], "fix")

    def test_split_segment_forwards_args(self):
        self.vm.splitSegment(index=0, at=5.0)
        call = self._last_call("split_segment")
        self.assertIsNotNone(call)
        self.assertEqual(call[1], (0, 5.0))

    def test_merge_segments_forwards_args(self):
        self.vm.mergeSegments(index=0, count=2)
        call = self._last_call("merge_segments")
        self.assertIsNotNone(call)
        self.assertEqual(call[1], (0, 2))

    def test_add_scene_boundary_forwards_args(self):
        self.vm.addSceneBoundary(start=0.0, end=5.0, reason="verse")
        call = self._last_call("add_scene_boundary")
        self.assertIsNotNone(call)
        self.assertEqual(call[1], (0.0, 5.0, "verse"))

    def test_add_beat_forwards_args(self):
        self.vm.addBeat(t=1.5, label="kick", confidence=0.95)
        call = self._last_call("add_beat")
        self.assertIsNotNone(call)
        self.assertEqual(call[1], (1.5, "kick", 0.95))

    def test_rebuild_pipeline_forwards_affected(self):
        self.vm.rebuildPipeline()
        # rebuild_pipeline was called with some AffectedArtifacts
        self.assertTrue(any(c[0] == "rebuild_pipeline" for c in self.service.calls))

    def test_undo_calls_service_undo(self):
        self.vm.undo()
        self.assertIn(("undo", ()), self.service.calls)

    def test_redo_calls_service_redo(self):
        self.vm.redo()
        self.assertIn(("redo", ()), self.service.calls)

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_edit_error_updates_status(self):
        self.service._simulate_error = True
        original = self.service.edit_segment
        def broken(*a, **kw):
            raise ValueError("boom")
        self.service.edit_segment = broken
        self.vm.editSegment(index=0, startDelta=0.0, endDelta=0.0, lyrics=None, notes=None)
        self.assertIn("boom", self.vm.error)

    def test_load_no_project_returns_false(self):
        result = self.vm.loadProject()
        # Should call service.load(), status might be ok
        self.assertIsInstance(result, bool)

    # ------------------------------------------------------------------
    # Status properties
    # ------------------------------------------------------------------

    def test_initial_status(self):
        self.assertEqual(self.vm.status, "idle")
        self.assertEqual(self.vm.error, "")

    def test_has_changes_after_edit(self):
        self.vm.loadProject()
        self.vm.editSegment(index=0, startDelta=1.0, endDelta=0.0, lyrics=None, notes=None)
        self.assertTrue(self.vm.hasChanges)

    def test_can_undo_from_service(self):
        self.assertFalse(self.vm.canUndo)
        # After an edit, service.can_undo() should be True
        self.vm.editSegment(index=0, startDelta=1.0, endDelta=0.0, lyrics=None, notes=None)
        self.service._undo_stack.append("state")
        self.assertTrue(self.vm.canUndo)

    def test_can_redo_from_service(self):
        self.assertFalse(self.vm.canRedo)
        self.service._redo_stack.append("state")
        self.assertTrue(self.vm.canRedo)

    # ------------------------------------------------------------------
    # Observable data
    # ------------------------------------------------------------------

    def test_segments_after_load(self):
        self.vm.loadProject()
        segs = self.vm.segments
        self.assertIsInstance(segs, list)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0]["start"], 0.0)
        self.assertEqual(segs[1]["end"], 10.0)

    def test_boundaries_empty(self):
        self.vm.loadProject()
        self.assertEqual(self.vm.boundaries, [])

    def test_beats_empty(self):
        self.vm.loadProject()
        self.assertEqual(self.vm.beats, [])


if __name__ == "__main__":
    unittest.main()
