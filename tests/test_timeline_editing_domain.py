from __future__ import annotations

import unittest

from feverslop.domain.timeline_editing import (
    BeatMarker,
    EditableTimelineSegment,
    SceneBoundary,
    TimelineEditImpact,
    TimelineSnapshot,
    compute_edit_impact,
    merge_segments,
    split_segment,
    validate_beat_markers,
    validate_scene_boundaries,
)


class EditableTimelineSegmentTest(unittest.TestCase):
    def test_create_from_timeline_segment(self):
        segment = EditableTimelineSegment(
            start=0.0, end=5.0, kind="vocals", text="hello"
        )
        self.assertEqual(segment.start, 0.0)
        self.assertEqual(segment.end, 5.0)
        self.assertEqual(segment.kind, "vocals")
        self.assertEqual(segment.text, "hello")
        self.assertIsNone(segment.lyrics_line)
        self.assertIsNone(segment.notes)
        self.assertFalse(segment.is_draft)

    def test_create_with_extra_fields(self):
        segment = EditableTimelineSegment(
            start=0.0,
            end=5.0,
            kind="vocals",
            text="hello",
            lyrics_line="Hello world",
            notes="Fix timing",
            is_draft=True,
        )
        self.assertEqual(segment.lyrics_line, "Hello world")
        self.assertEqual(segment.notes, "Fix timing")
        self.assertTrue(segment.is_draft)

    def test_frozen(self):
        segment = EditableTimelineSegment(
            start=0.0, end=5.0, kind="vocals", text="hello"
        )
        with self.assertRaises(Exception):
            segment.start = 1.0  # type: ignore

    def test_rejects_negative_start(self):
        with self.assertRaises(ValueError):
            EditableTimelineSegment(start=-1.0, end=5.0, kind="vocals", text="hi")

    def test_rejects_negative_end(self):
        with self.assertRaises(ValueError):
            EditableTimelineSegment(start=0.0, end=-1.0, kind="vocals", text="hi")

    def test_rejects_end_before_start(self):
        with self.assertRaises(ValueError):
            EditableTimelineSegment(start=5.0, end=2.0, kind="vocals", text="hi")

    def test_allows_zero_length(self):
        segment = EditableTimelineSegment(start=0.0, end=0.0, kind="instrumental")
        self.assertEqual(segment.start, 0.0)
        self.assertEqual(segment.end, 0.0)


class SceneBoundaryTest(unittest.TestCase):
    def test_valid_boundary(self):
        boundary = SceneBoundary(start=0.0, end=5.0, reason="verse starts")
        self.assertEqual(boundary.start, 0.0)
        self.assertEqual(boundary.end, 5.0)
        self.assertEqual(boundary.reason, "verse starts")
        self.assertEqual(boundary.min_duration, 2.0)

    def test_custom_min_duration(self):
        boundary = SceneBoundary(
            start=0.0, end=10.0, reason="long scene", min_duration=8.0
        )
        self.assertEqual(boundary.min_duration, 8.0)

    def test_rejects_too_short(self):
        with self.assertRaises(ValueError):
            SceneBoundary(start=0.0, end=1.0, reason="too short")

    def test_rejects_negative_start(self):
        with self.assertRaises(ValueError):
            SceneBoundary(start=-1.0, end=5.0, reason="bad")

    def test_rejects_negative_end(self):
        with self.assertRaises(ValueError):
            SceneBoundary(start=0.0, end=-1.0, reason="bad")

    def test_rejects_end_before_start(self):
        with self.assertRaises(ValueError):
            SceneBoundary(start=5.0, end=2.0, reason="bad")

    def test_frozen(self):
        boundary = SceneBoundary(start=0.0, end=5.0, reason="verse")
        with self.assertRaises(Exception):
            boundary.start = 1.0  # type: ignore


class BeatMarkerTest(unittest.TestCase):
    def test_valid_marker(self):
        marker = BeatMarker(time_s=1.5, label="kick", confidence=0.95)
        self.assertEqual(marker.time_s, 1.5)
        self.assertEqual(marker.label, "kick")
        self.assertEqual(marker.confidence, 0.95)

    def test_rejects_negative_time(self):
        with self.assertRaises(ValueError):
            BeatMarker(time_s=-1.0, label="bad", confidence=0.5)

    def test_rejects_invalid_confidence(self):
        with self.assertRaises(ValueError):
            BeatMarker(time_s=1.0, label="bad", confidence=1.5)
        with self.assertRaises(ValueError):
            BeatMarker(time_s=1.0, label="bad", confidence=-0.1)

    def test_frozen(self):
        marker = BeatMarker(time_s=1.0, label="snare", confidence=0.8)
        with self.assertRaises(Exception):
            marker.time_s = 2.0  # type: ignore


class TimelineSnapshotTest(unittest.TestCase):
    def _make_segment(self, start: float, end: float) -> EditableTimelineSegment:
        return EditableTimelineSegment(
            start=start, end=end, kind="vocals", text="test"
        )

    def _make_boundary(self, start: float, end: float) -> SceneBoundary:
        return SceneBoundary(start=start, end=end, reason="test")

    def _make_marker(self, time_s: float) -> BeatMarker:
        return BeatMarker(time_s=time_s, label="beat", confidence=0.9)

    def test_create_snapshot(self):
        snapshot = TimelineSnapshot(
            segments=[self._make_segment(0, 5)],
            scene_boundaries=[self._make_boundary(0, 5)],
            beat_markers=[self._make_marker(1.0)],
            metadata={"version": 1},
        )
        self.assertEqual(len(snapshot.segments), 1)
        self.assertEqual(len(snapshot.scene_boundaries), 1)
        self.assertEqual(len(snapshot.beat_markers), 1)
        self.assertEqual(snapshot.metadata["version"], 1)

    def test_empty_snapshot(self):
        snapshot = TimelineSnapshot(
            segments=[],
            scene_boundaries=[],
            beat_markers=[],
            metadata={},
        )
        self.assertEqual(len(snapshot.segments), 0)

    def test_to_json_roundtrip(self):
        original = TimelineSnapshot(
            segments=[self._make_segment(0, 5), self._make_segment(5, 10)],
            scene_boundaries=[self._make_boundary(0, 5)],
            beat_markers=[self._make_marker(1.0), self._make_marker(3.5)],
            metadata={"version": 1},
        )
        data = original.to_json()
        restored = TimelineSnapshot.from_json(data)
        self.assertEqual(len(restored.segments), 2)
        self.assertEqual(restored.segments[0].start, 0.0)
        self.assertEqual(restored.segments[0].end, 5.0)
        self.assertEqual(restored.segments[1].start, 5.0)
        self.assertEqual(restored.segments[1].end, 10.0)
        self.assertEqual(len(restored.scene_boundaries), 1)
        self.assertEqual(restored.scene_boundaries[0].reason, "test")
        self.assertEqual(len(restored.beat_markers), 2)
        self.assertEqual(restored.beat_markers[0].time_s, 1.0)
        self.assertEqual(restored.beat_markers[1].time_s, 3.5)
        self.assertEqual(restored.metadata["version"], 1)

    def test_to_json_produces_dict(self):
        snapshot = TimelineSnapshot(
            segments=[], scene_boundaries=[], beat_markers=[], metadata={}
        )
        data = snapshot.to_json()
        self.assertIsInstance(data, dict)

    def test_json_contains_required_keys(self):
        snapshot = TimelineSnapshot(
            segments=[], scene_boundaries=[], beat_markers=[], metadata={}
        )
        data = snapshot.to_json()
        for key in ("segments", "scene_boundaries", "beat_markers", "metadata"):
            self.assertIn(key, data)

    def test_snapshot_frozen(self):
        snapshot = TimelineSnapshot(
            segments=[], scene_boundaries=[], beat_markers=[], metadata={}
        )
        with self.assertRaises(Exception):
            snapshot.metadata["x"] = 1  # type: ignore


class TimelineEditImpactTest(unittest.TestCase):
    def test_defaults_to_no_invalidation(self):
        impact = TimelineEditImpact()
        self.assertFalse(impact.timeline_invalidated)
        self.assertFalse(impact.scene_srt_invalidated)
        self.assertFalse(impact.beat_json_invalidated)
        self.assertFalse(impact.stage1_segments_invalidated)
        self.assertFalse(impact.ltx_prompt_invalidated)
        self.assertFalse(impact.render_plan_invalidated)

    def test_can_set_flags(self):
        impact = TimelineEditImpact(timeline_invalidated=True, render_plan_invalidated=True)
        self.assertTrue(impact.timeline_invalidated)
        self.assertTrue(impact.render_plan_invalidated)
        self.assertFalse(impact.beat_json_invalidated)

    def test_frozen(self):
        impact = TimelineEditImpact(timeline_invalidated=True)
        with self.assertRaises(Exception):
            impact.timeline_invalidated = False  # type: ignore


class SplitSegmentTest(unittest.TestCase):
    def split_at(self, start: float, end: float, at: float):
        seg = EditableTimelineSegment(
            start=start, end=end, kind="vocals", text="hello world"
        )
        return split_segment(seg, at)

    def test_splits_at_midpoint(self):
        left, right = self.split_at(0.0, 10.0, 5.0)
        self.assertEqual(left.start, 0.0)
        self.assertEqual(left.end, 5.0)
        self.assertEqual(right.start, 5.0)
        self.assertEqual(right.end, 10.0)

    def test_preserves_total_duration(self):
        left, right = self.split_at(0.0, 10.0, 5.0)
        total = (left.end - left.start) + (right.end - right.start)
        self.assertAlmostEqual(total, 10.0)

    def test_preserves_kind_and_text(self):
        left, right = self.split_at(0.0, 10.0, 5.0)
        self.assertEqual(left.kind, "vocals")
        self.assertEqual(right.kind, "vocals")
        self.assertEqual(left.text, "hello world")
        self.assertEqual(right.text, "hello world")

    def test_preserves_lyrics_line(self):
        seg = EditableTimelineSegment(
            start=0.0, end=10.0, kind="vocals", text="", lyrics_line="la la"
        )
        left, right = split_segment(seg, 5.0)
        self.assertEqual(left.lyrics_line, "la la")
        self.assertEqual(right.lyrics_line, "la la")

    def test_preserves_notes_and_draft(self):
        seg = EditableTimelineSegment(
            start=0.0, end=10.0, kind="vocals", text="",
            notes="fix this", is_draft=True,
        )
        left, right = split_segment(seg, 5.0)
        self.assertEqual(left.notes, "fix this")
        self.assertTrue(left.is_draft)
        self.assertEqual(right.notes, "fix this")
        self.assertTrue(right.is_draft)

    def test_raises_when_at_equals_start(self):
        with self.assertRaises(ValueError):
            self.split_at(5.0, 10.0, 5.0)

    def test_raises_when_at_equals_end(self):
        with self.assertRaises(ValueError):
            self.split_at(0.0, 10.0, 10.0)

    def test_raises_when_at_before_start(self):
        with self.assertRaises(ValueError):
            self.split_at(5.0, 10.0, 3.0)

    def test_raises_when_at_after_end(self):
        with self.assertRaises(ValueError):
            self.split_at(0.0, 10.0, 12.0)

    def test_splits_instrumental_segment(self):
        seg = EditableTimelineSegment(start=0.0, end=8.0, kind="instrumental")
        left, right = split_segment(seg, 3.0)
        self.assertEqual(left.kind, "instrumental")
        self.assertEqual(right.kind, "instrumental")


class MergeSegmentsTest(unittest.TestCase):
    def merge_all(self, segments: list[EditableTimelineSegment]):
        return merge_segments(segments)

    def test_merges_adjacent_segments(self):
        segs = [
            EditableTimelineSegment(start=0.0, end=5.0, kind="vocals", text="hello"),
            EditableTimelineSegment(start=5.0, end=10.0, kind="vocals", text="world"),
        ]
        merged = self.merge_all(segs)
        self.assertEqual(merged.start, 0.0)
        self.assertEqual(merged.end, 10.0)

    def test_preserves_total_duration(self):
        segs = [
            EditableTimelineSegment(start=0.0, end=5.0, kind="vocals", text="hello"),
            EditableTimelineSegment(start=5.0, end=10.0, kind="vocals", text="world"),
        ]
        merged = self.merge_all(segs)
        original_total = sum(s.end - s.start for s in segs)
        self.assertAlmostEqual(merged.end - merged.start, original_total)

    def test_merges_texts(self):
        segs = [
            EditableTimelineSegment(start=0.0, end=5.0, kind="vocals", text="hello"),
            EditableTimelineSegment(start=5.0, end=10.0, kind="vocals", text="world"),
        ]
        merged = self.merge_all(segs)
        self.assertEqual(merged.text, "hello world")

    def test_raises_empty_list(self):
        with self.assertRaises(ValueError):
            self.merge_all([])

    def test_raises_single_segment(self):
        seg = EditableTimelineSegment(start=0.0, end=5.0, kind="vocals")
        with self.assertRaises(ValueError):
            self.merge_all([seg])

    def test_raises_gap_between_segments(self):
        segs = [
            EditableTimelineSegment(start=0.0, end=5.0, kind="vocals", text="a"),
            EditableTimelineSegment(start=6.0, end=10.0, kind="vocals", text="b"),
        ]
        with self.assertRaises(ValueError):
            self.merge_all(segs)

    def test_raises_overlap(self):
        segs = [
            EditableTimelineSegment(start=0.0, end=6.0, kind="vocals", text="a"),
            EditableTimelineSegment(start=5.0, end=10.0, kind="vocals", text="b"),
        ]
        with self.assertRaises(ValueError):
            self.merge_all(segs)

    def test_merges_three_segments(self):
        segs = [
            EditableTimelineSegment(start=0.0, end=3.0, kind="vocals", text="a"),
            EditableTimelineSegment(start=3.0, end=6.0, kind="vocals", text="b"),
            EditableTimelineSegment(start=6.0, end=9.0, kind="vocals", text="c"),
        ]
        merged = self.merge_all(segs)
        self.assertEqual(merged.start, 0.0)
        self.assertEqual(merged.end, 9.0)
        self.assertEqual(merged.text, "a b c")


class ValidateSceneBoundariesTest(unittest.TestCase):
    def _bnd(self, start: float, end: float, reason: str = "test") -> SceneBoundary:
        return SceneBoundary(start=start, end=end, reason=reason)

    def test_sorts_by_start_time(self):
        boundaries = [
            self._bnd(10.0, 15.0, "second"),
            self._bnd(0.0, 5.0, "first"),
        ]
        result = validate_scene_boundaries(boundaries)
        self.assertEqual(result[0].reason, "first")
        self.assertEqual(result[1].reason, "second")

    def test_rejects_overlapping_boundaries(self):
        boundaries = [
            self._bnd(0.0, 10.0, "first"),
            self._bnd(8.0, 15.0, "second"),
        ]
        with self.assertRaises(ValueError):
            validate_scene_boundaries(boundaries)

    def test_accepts_adjacent_boundaries(self):
        boundaries = [
            self._bnd(0.0, 5.0, "first"),
            self._bnd(5.0, 10.0, "second"),
        ]
        result = validate_scene_boundaries(boundaries)
        self.assertEqual(len(result), 2)

    def test_empty_boundaries(self):
        result = validate_scene_boundaries([])
        self.assertEqual(result, [])

    def test_single_boundary(self):
        boundaries = [self._bnd(0.0, 5.0, "only")]
        result = validate_scene_boundaries(boundaries)
        self.assertEqual(len(result), 1)

    def test_rejects_unsorted_then_valid_overlap(self):
        """Even if input is unsorted, must detect overlaps after sorting."""
        boundaries = [
            self._bnd(5.0, 12.0, "b"),
            self._bnd(0.0, 8.0, "a"),
        ]
        with self.assertRaises(ValueError):
            validate_scene_boundaries(boundaries)


class ValidateBeatMarkersTest(unittest.TestCase):
    def _mk(self, time_s: float, label: str = "beat", confidence: float = 0.9) -> BeatMarker:
        return BeatMarker(time_s=time_s, label=label, confidence=confidence)

    def test_sorts_by_time(self):
        markers = [
            self._mk(5.0, "b"),
            self._mk(1.0, "a"),
            self._mk(3.0, "c"),
        ]
        result = validate_beat_markers(markers)
        self.assertEqual(result[0].time_s, 1.0)
        self.assertEqual(result[1].time_s, 3.0)
        self.assertEqual(result[2].time_s, 5.0)

    def test_rejects_duplicate_times(self):
        markers = [
            self._mk(1.0, "first"),
            self._mk(1.0, "second"),
        ]
        with self.assertRaises(ValueError):
            validate_beat_markers(markers)

    def test_empty_markers(self):
        result = validate_beat_markers([])
        self.assertEqual(result, [])

    def test_single_marker(self):
        markers = [self._mk(1.0)]
        result = validate_beat_markers(markers)
        self.assertEqual(len(result), 1)

    def test_near_duplicate_tolerance(self):
        """Markers within epsilon are treated as duplicates."""
        markers = [
            self._mk(1.0, "first"),
            self._mk(1.0000001, "second"),
        ]
        with self.assertRaises(ValueError):
            validate_beat_markers(markers)


class ComputeEditImpactTest(unittest.TestCase):
    def _snapshot(
        self,
        segments: list[EditableTimelineSegment] | None = None,
        boundaries: list[SceneBoundary] | None = None,
        markers: list[BeatMarker] | None = None,
        meta: dict | None = None,
    ) -> TimelineSnapshot:
        return TimelineSnapshot(
            segments=segments or [],
            scene_boundaries=boundaries or [],
            beat_markers=markers or [],
            metadata=meta or {},
        )

    def _seg(self, start: float, end: float, **kw) -> EditableTimelineSegment:
        return EditableTimelineSegment(
            start=start, end=end, kind=kw.pop("kind", "vocals"),
            text=kw.pop("text", "test"), **kw
        )

    def _bnd(self, start: float, end: float) -> SceneBoundary:
        return SceneBoundary(start=start, end=end, reason="test")

    def _mk(self, time_s: float) -> BeatMarker:
        return BeatMarker(time_s=time_s, label="beat", confidence=0.9)

    def test_no_change_no_invalidation(self):
        segs = [self._seg(0, 5)]
        s = self._snapshot(segments=segs)
        impact = compute_edit_impact(s, s)
        self.assertFalse(impact.timeline_invalidated)
        self.assertFalse(impact.scene_srt_invalidated)
        self.assertFalse(impact.beat_json_invalidated)
        self.assertFalse(impact.stage1_segments_invalidated)
        self.assertFalse(impact.ltx_prompt_invalidated)
        self.assertFalse(impact.render_plan_invalidated)

    def test_segment_count_change_invalidates_timeline(self):
        before = self._snapshot(segments=[self._seg(0, 5)])
        after = self._snapshot(segments=[self._seg(0, 3), self._seg(3, 5)])
        impact = compute_edit_impact(before, after)
        self.assertTrue(impact.timeline_invalidated)
        self.assertTrue(impact.stage1_segments_invalidated)
        self.assertTrue(impact.render_plan_invalidated)

    def test_segment_time_change_invalidates_timeline(self):
        before = self._snapshot(segments=[self._seg(0, 5)])
        after = self._snapshot(segments=[self._seg(1, 6)])
        impact = compute_edit_impact(before, after)
        self.assertTrue(impact.timeline_invalidated)

    def test_boundary_change_invalidates_scene_srt(self):
        before = self._snapshot(boundaries=[self._bnd(0, 5)])
        after = self._snapshot(boundaries=[self._bnd(0, 3), self._bnd(3, 5)])
        impact = compute_edit_impact(before, after)
        self.assertTrue(impact.scene_srt_invalidated)
        self.assertTrue(impact.render_plan_invalidated)

    def test_beat_marker_change_invalidates_beat_json(self):
        before = self._snapshot(markers=[self._mk(1.0)])
        after = self._snapshot(markers=[self._mk(1.0), self._mk(2.0)])
        impact = compute_edit_impact(before, after)
        self.assertTrue(impact.beat_json_invalidated)

    def test_text_change_invalidates_ltx_prompt(self):
        before = self._snapshot(segments=[self._seg(0, 5, text="hello")])
        after = self._snapshot(segments=[self._seg(0, 5, text="goodbye")])
        impact = compute_edit_impact(before, after)
        self.assertTrue(impact.ltx_prompt_invalidated)
        self.assertTrue(impact.render_plan_invalidated)

    def test_lyrics_change_invalidates_ltx_prompt(self):
        before = self._snapshot(segments=[self._seg(0, 5, lyrics_line="la")])
        after = self._snapshot(segments=[self._seg(0, 5, lyrics_line="lu")])
        impact = compute_edit_impact(before, after)
        self.assertTrue(impact.ltx_prompt_invalidated)

    def test_metadata_change_invalidates_render_plan(self):
        before = self._snapshot(meta={"version": "1"})
        after = self._snapshot(meta={"version": "2"})
        impact = compute_edit_impact(before, after)
        self.assertTrue(impact.render_plan_invalidated)

    def test_kind_change_invalidates_stage1(self):
        before = self._snapshot(segments=[self._seg(0, 5)])
        after = self._snapshot(segments=[EditableTimelineSegment(start=0, end=5, kind="instrumental")])
        impact = compute_edit_impact(before, after)
        self.assertTrue(impact.stage1_segments_invalidated)


if __name__ == "__main__":
    unittest.main()
