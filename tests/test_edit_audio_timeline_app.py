from __future__ import annotations

import copy
import unittest
from typing import Any

from feverslop.domain.timeline_editing import (
    BeatMarker,
    EditableTimelineSegment,
    SceneBoundary,
    TimelineSnapshot,
)
from feverslop.ports.timeline_documents import (
    AffectedArtifacts,
)


# ---------------------------------------------------------------------------
# Fake port implementations (shared across tests)
# ---------------------------------------------------------------------------


class FakeReadPort:
    """In-memory TimelineReadPort for tests."""

    def __init__(self) -> None:
        self.timeline: list[dict[str, Any]] = []
        self.scene_srt: str | None = None
        self.beat_json: list[dict[str, Any]] | None = None
        self.stage1_segments: list[dict[str, Any]] | None = None
        self.ltx_prompt_relay: list[dict[str, Any]] | None = None
        self.render_plan: dict[str, Any] | None = None

    def read_timeline(self) -> list[dict]:
        return copy.deepcopy(self.timeline)

    def read_scene_srt(self) -> str | None:
        return self.scene_srt

    def read_beat_json(self) -> list[dict] | None:
        return copy.deepcopy(self.beat_json) if self.beat_json is not None else None

    def read_stage1_segments(self) -> list[dict] | None:
        return copy.deepcopy(self.stage1_segments) if self.stage1_segments is not None else None

    def read_ltx_prompt_relay(self) -> list[dict] | None:
        return copy.deepcopy(self.ltx_prompt_relay) if self.ltx_prompt_relay is not None else None

    def read_render_plan(self) -> dict | None:
        return copy.deepcopy(self.render_plan) if self.render_plan is not None else None


class FakeWritePort:
    """In-memory TimelineWritePort for tests."""

    def __init__(self, raise_on_write: bool = False) -> None:
        self.timeline: list[dict[str, Any]] | None = None
        self.scene_srt: str | None = None
        self.raise_on_write = raise_on_write

    def write_timeline(self, data: list[dict]) -> None:
        if self.raise_on_write:
            raise OSError("disk full")
        self.timeline = copy.deepcopy(data)

    def write_scene_srt(self, content: str) -> None:
        if self.raise_on_write:
            raise OSError("disk full")
        self.scene_srt = content


# ---------------------------------------------------------------------------
# Build a sample timeline dict list → TimelineSnapshot helpers
# ---------------------------------------------------------------------------


def _make_timeline_dicts(
    *,
    segments: list[EditableTimelineSegment] | None = None,
    boundaries: list[SceneBoundary] | None = None,
    markers: list[BeatMarker] | None = None,
    meta: dict | None = None,
) -> list[dict]:
    snap = TimelineSnapshot(
        segments=segments or [],
        scene_boundaries=boundaries or [],
        beat_markers=markers or [],
        metadata=meta or {},
    )
    return [snap.to_json()]


def _make_seg(start: float, end: float, **kw: Any) -> EditableTimelineSegment:
    return EditableTimelineSegment(
        start=start, end=end, kind=kw.pop("kind", "vocals"),
        text=kw.pop("text", "test"), **kw
    )


def _make_bnd(start: float, end: float) -> SceneBoundary:
    return SceneBoundary(start=start, end=end, reason="test")


def _make_mk(time_s: float) -> BeatMarker:
    return BeatMarker(time_s=time_s, label="beat", confidence=0.9)


# ---------------------------------------------------------------------------
# ComputeEditImpact tests (application layer wrapper around domain function)
# ---------------------------------------------------------------------------


class ComputeEditImpactTest(unittest.TestCase):
    """Test the application-layer ComputeEditImpact that wraps domain compute_edit_impact."""

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

    def test_no_change_yields_no_impact(self):
        snap = self._snapshot(segments=[_make_seg(0, 5)])
        from feverslop.application.edit_audio_timeline import ComputeEditImpact
        impact = ComputeEditImpact(snap, snap)
        self.assertFalse(impact.timeline)
        self.assertFalse(impact.scene_srt)
        self.assertFalse(impact.beat_json)
        self.assertFalse(impact.stage1_segments)
        self.assertFalse(impact.ltx_prompt)
        self.assertFalse(impact.render_plan)

    def test_segment_trim_invalidates_all(self):
        """Splitting a segment (count change) should invalidate everything downstream."""
        before = self._snapshot(segments=[_make_seg(0, 10)])
        after = self._snapshot(segments=[_make_seg(0, 5), _make_seg(5, 10)])
        from feverslop.application.edit_audio_timeline import ComputeEditImpact
        impact = ComputeEditImpact(before, after)
        # render plan is invalidated because segment count changed -> text change propagates
        self.assertTrue(impact.timeline)
        self.assertTrue(impact.stage1_segments)
        self.assertTrue(impact.render_plan)

    def test_boundary_change_invalidates_scene_only(self):
        """Changing only scene boundaries should hit scene_srt and render_plan."""
        before = self._snapshot(boundaries=[_make_bnd(0, 5)])
        after = self._snapshot(boundaries=[_make_bnd(0, 3), _make_bnd(3, 5)])
        from feverslop.application.edit_audio_timeline import ComputeEditImpact
        impact = ComputeEditImpact(before, after)
        self.assertTrue(impact.scene_srt)
        self.assertTrue(impact.render_plan)
        self.assertFalse(impact.beat_json)
        self.assertFalse(impact.stage1_segments)
        self.assertFalse(impact.ltx_prompt)

    def test_beat_change_invalidates_beat_stage1_prompt(self):
        """Adding a beat marker should invalidate beat_json only (not stage1 or prompt)."""
        before = self._snapshot(markers=[_make_mk(1.0)])
        after = self._snapshot(markers=[_make_mk(1.0), _make_mk(2.0)])
        from feverslop.application.edit_audio_timeline import ComputeEditImpact
        impact = ComputeEditImpact(before, after)
        self.assertTrue(impact.beat_json)
        self.assertFalse(impact.timeline)
        self.assertFalse(impact.stage1_segments)
        self.assertFalse(impact.ltx_prompt)
        self.assertFalse(impact.scene_srt)
        self.assertFalse(impact.render_plan)

    def test_text_change_invalidates_ltx_and_render_plan(self):
        before = self._snapshot(segments=[_make_seg(0, 5, text="hello")])
        after = self._snapshot(segments=[_make_seg(0, 5, text="goodbye")])
        from feverslop.application.edit_audio_timeline import ComputeEditImpact
        impact = ComputeEditImpact(before, after)
        self.assertTrue(impact.ltx_prompt)
        self.assertTrue(impact.render_plan)
        self.assertTrue(impact.timeline)
        self.assertFalse(impact.scene_srt)

    def test_lyrics_change_invalidates_ltx_only(self):
        before = self._snapshot(segments=[_make_seg(0, 5, lyrics_line="la")])
        after = self._snapshot(segments=[_make_seg(0, 5, lyrics_line="lu")])
        from feverslop.application.edit_audio_timeline import ComputeEditImpact
        impact = ComputeEditImpact(before, after)
        self.assertTrue(impact.ltx_prompt)
        self.assertTrue(impact.timeline)
        self.assertFalse(impact.scene_srt)
        self.assertFalse(impact.beat_json)

    def test_metadata_change_invalidates_render_plan(self):
        before = self._snapshot(meta={"version": "1"})
        after = self._snapshot(meta={"version": "2"})
        from feverslop.application.edit_audio_timeline import ComputeEditImpact
        impact = ComputeEditImpact(before, after)
        self.assertTrue(impact.render_plan)
        self.assertFalse(impact.timeline)


# ---------------------------------------------------------------------------
# SaveTimelines tests
# ---------------------------------------------------------------------------


class SaveTimelinesTest(unittest.TestCase):
    """SaveTimelines orchestrates read current state + write via write_port."""

    def test_saves_timeline_and_srt(self):
        from feverslop.application.edit_audio_timeline import SaveTimelines
        read_port = FakeReadPort()
        write_port = FakeWritePort()

        segs = [_make_seg(0, 5)]
        snap = TimelineSnapshot(segments=segs, scene_boundaries=[], beat_markers=[], metadata={})
        read_port.timeline = [snap.to_json()]
        read_port.scene_srt = "1\n0:00:00 --> 0:00:05\nhello"

        SaveTimelines(read_port, write_port).execute()

        self.assertIsNotNone(write_port.timeline)
        self.assertEqual(write_port.timeline[0]["segments"][0]["start"], 0.0)
        self.assertEqual(write_port.scene_srt, "1\n0:00:00 --> 0:00:05\nhello")

    def test_empty_timeline_is_saved(self):
        from feverslop.application.edit_audio_timeline import SaveTimelines
        read_port = FakeReadPort()
        write_port = FakeWritePort()
        read_port.timeline = []

        SaveTimelines(read_port, write_port).execute()

        self.assertEqual(write_port.timeline, [])

    def test_write_error_surfaces(self):
        from feverslop.application.edit_audio_timeline import SaveTimelines
        read_port = FakeReadPort()
        write_port = FakeWritePort(raise_on_write=True)
        read_port.timeline = [_make_timeline_dicts(segments=[_make_seg(0, 5)])[0]]

        with self.assertRaises(OSError):
            SaveTimelines(read_port, write_port).execute()

    def test_no_scene_srt_does_not_write_srt(self):
        from feverslop.application.edit_audio_timeline import SaveTimelines
        read_port = FakeReadPort()
        write_port = FakeWritePort()
        read_port.timeline = []
        read_port.scene_srt = None

        SaveTimelines(read_port, write_port).execute()

        self.assertIsNone(write_port.scene_srt)


# ---------------------------------------------------------------------------
# RebuildDownstreamArtifacts tests
# ---------------------------------------------------------------------------


class RebuildDownstreamArtifactsTest(unittest.TestCase):
    """RebuildDownstreamArtifacts returns pipeline job keys for invalidated artifacts."""

    def test_no_impact_returns_empty(self):
        from feverslop.application.edit_audio_timeline import RebuildDownstreamArtifacts
        impact = AffectedArtifacts()
        jobs = RebuildDownstreamArtifacts(impact)
        self.assertEqual(jobs, [])

    def test_timeline_flag_triggers_timeline_key(self):
        from feverslop.application.edit_audio_timeline import RebuildDownstreamArtifacts
        impact = AffectedArtifacts(timeline=True)
        jobs = RebuildDownstreamArtifacts(impact)
        self.assertIn("timeline", jobs)

    def test_scene_srt_flag_triggers_scene_srt_key(self):
        from feverslop.application.edit_audio_timeline import RebuildDownstreamArtifacts
        impact = AffectedArtifacts(scene_srt=True)
        jobs = RebuildDownstreamArtifacts(impact)
        self.assertIn("scene_srt", jobs)

    def test_beat_json_flag_triggers_beat_key(self):
        from feverslop.application.edit_audio_timeline import RebuildDownstreamArtifacts
        impact = AffectedArtifacts(beat_json=True)
        jobs = RebuildDownstreamArtifacts(impact)
        self.assertIn("beat_json", jobs)

    def test_stage1_flag_triggers_stage1_key(self):
        from feverslop.application.edit_audio_timeline import RebuildDownstreamArtifacts
        impact = AffectedArtifacts(stage1_segments=True)
        jobs = RebuildDownstreamArtifacts(impact)
        self.assertIn("stage1_segments", jobs)

    def test_ltx_prompt_flag_triggers_prompt_key(self):
        from feverslop.application.edit_audio_timeline import RebuildDownstreamArtifacts
        impact = AffectedArtifacts(ltx_prompt=True)
        jobs = RebuildDownstreamArtifacts(impact)
        self.assertIn("ltx_prompt", jobs)

    def test_render_plan_flag_triggers_render_plan_key(self):
        from feverslop.application.edit_audio_timeline import RebuildDownstreamArtifacts
        impact = AffectedArtifacts(render_plan=True)
        jobs = RebuildDownstreamArtifacts(impact)
        self.assertIn("render_plan", jobs)

    def test_multiple_flags_return_all_keys(self):
        from feverslop.application.edit_audio_timeline import RebuildDownstreamArtifacts
        impact = AffectedArtifacts(
            timeline=True,
            scene_srt=True,
            beat_json=True,
            stage1_segments=True,
            ltx_prompt=True,
            render_plan=True,
        )
        jobs = RebuildDownstreamArtifacts(impact)
        self.assertEqual(sorted(jobs), sorted([
            "timeline", "scene_srt", "beat_json",
            "stage1_segments", "ltx_prompt", "render_plan",
        ]))

    def test_segment_split_triggers_downstream(self):
        """A segment split (count change) triggers stage1 and render_plan jobs."""
        from feverslop.application.edit_audio_timeline import ComputeEditImpact, RebuildDownstreamArtifacts
        before = TimelineSnapshot(
            segments=[_make_seg(0, 10)],
            scene_boundaries=[],
            beat_markers=[],
            metadata={},
        )
        after = TimelineSnapshot(
            segments=[_make_seg(0, 5), _make_seg(5, 10)],
            scene_boundaries=[],
            beat_markers=[],
            metadata={},
        )
        impact = ComputeEditImpact(before, after)
        jobs = RebuildDownstreamArtifacts(impact)
        self.assertIn("stage1_segments", jobs)
        self.assertIn("render_plan", jobs)


# ---------------------------------------------------------------------------
# EditAudioTimeline use case orchestration tests
# ---------------------------------------------------------------------------


class EditAudioTimelineTest(unittest.TestCase):
    """EditAudioTimeline orchestrates: read → validate → write → compute impact."""

    def setUp(self):
        self.read_port = FakeReadPort()
        self.write_port = FakeWritePort()

    def test_happy_path_edit(self):
        from feverslop.application.edit_audio_timeline import EditAudioTimeline, EditResult
        segs = [_make_seg(0, 10)]
        snap = TimelineSnapshot(segments=segs, scene_boundaries=[], beat_markers=[], metadata={})
        self.read_port.timeline = [snap.to_json()]

        use_case = EditAudioTimeline(
            read_port=self.read_port,
            write_port=self.write_port,
        )
        result = use_case.edit(segment_index=0, changes={"start": 0.0, "end": 10.0})

        self.assertIsInstance(result, EditResult)
        self.assertIsNotNone(result.after_snapshot)
        self.assertIsNotNone(result.impact)
        self.assertIsNotNone(result.timestamp)

    def test_edit_returns_impact(self):
        from feverslop.application.edit_audio_timeline import EditAudioTimeline
        segs = [_make_seg(0, 10)]
        snap = TimelineSnapshot(segments=segs, scene_boundaries=[], beat_markers=[], metadata={})
        self.read_port.timeline = [snap.to_json()]

        use_case = EditAudioTimeline(
            read_port=self.read_port,
            write_port=self.write_port,
        )
        # Changing nothing should produce no impact
        result = use_case.edit(segment_index=0, changes={"text": segs[0].text})
        self.assertFalse(result.impact.any())

    def test_validation_failure_surfaces(self):
        from feverslop.application.edit_audio_timeline import EditAudioTimeline, EditError
        segs = [_make_seg(0, 5)]
        snap = TimelineSnapshot(segments=segs, scene_boundaries=[], beat_markers=[], metadata={})
        self.read_port.timeline = [snap.to_json()]

        use_case = EditAudioTimeline(
            read_port=self.read_port,
            write_port=self.write_port,
        )
        # negative start should fail validation
        with self.assertRaises(EditError):
            use_case.edit(segment_index=0, changes={"start": -1.0})

    def test_write_error_surfaces(self):
        from feverslop.application.edit_audio_timeline import EditAudioTimeline, EditError
        segs = [_make_seg(0, 5)]
        snap = TimelineSnapshot(segments=segs, scene_boundaries=[], beat_markers=[], metadata={})
        self.read_port.timeline = [snap.to_json()]
        self.write_port = FakeWritePort(raise_on_write=True)

        use_case = EditAudioTimeline(
            read_port=self.read_port,
            write_port=self.write_port,
        )
        # Write port raises → EditError wrapping it
        with self.assertRaises(EditError):
            use_case.edit(segment_index=0, changes={"text": "modified"})

    def test_out_of_range_index_surfaces(self):
        from feverslop.application.edit_audio_timeline import EditAudioTimeline, EditError
        segs = [_make_seg(0, 5)]
        snap = TimelineSnapshot(segments=segs, scene_boundaries=[], beat_markers=[], metadata={})
        self.read_port.timeline = [snap.to_json()]

        use_case = EditAudioTimeline(
            read_port=self.read_port,
            write_port=self.write_port,
        )
        with self.assertRaises(EditError):
            use_case.edit(segment_index=99, changes={"text": "x"})

    def test_edit_writes_updated_timeline(self):
        from feverslop.application.edit_audio_timeline import EditAudioTimeline
        segs = [_make_seg(0, 5, text="hello")]
        snap = TimelineSnapshot(segments=segs, scene_boundaries=[], beat_markers=[], metadata={})
        self.read_port.timeline = [snap.to_json()]

        use_case = EditAudioTimeline(
            read_port=self.read_port,
            write_port=self.write_port,
        )
        use_case.edit(segment_index=0, changes={"text": "goodbye"})

        self.assertIsNotNone(self.write_port.timeline)
        self.assertEqual(self.write_port.timeline[0]["segments"][0]["text"], "goodbye")

    def test_edit_computes_before_and_after(self):
        from feverslop.application.edit_audio_timeline import EditAudioTimeline
        segs = [_make_seg(0, 5, text="hello")]
        snap = TimelineSnapshot(segments=segs, scene_boundaries=[], beat_markers=[], metadata={})
        self.read_port.timeline = [snap.to_json()]

        use_case = EditAudioTimeline(
            read_port=self.read_port,
            write_port=self.write_port,
        )
        result = use_case.edit(segment_index=0, changes={"text": "goodbye"})

        self.assertTrue(result.impact.timeline)
        self.assertTrue(result.impact.ltx_prompt)

    def test_timestamp_is_present(self):
        from feverslop.application.edit_audio_timeline import EditAudioTimeline
        segs = [_make_seg(0, 5)]
        snap = TimelineSnapshot(segments=segs, scene_boundaries=[], beat_markers=[], metadata={})
        self.read_port.timeline = [snap.to_json()]

        import datetime
        before = datetime.datetime.now(datetime.timezone.utc)
        use_case = EditAudioTimeline(
            read_port=self.read_port,
            write_port=self.write_port,
        )
        result = use_case.edit(segment_index=0, changes={"text": "x"})
        after = datetime.datetime.now(datetime.timezone.utc)

        self.assertIsNotNone(result.timestamp)
        self.assertTrue(before <= result.timestamp <= after)


if __name__ == "__main__":
    unittest.main()
