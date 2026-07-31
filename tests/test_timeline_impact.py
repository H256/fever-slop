"""Application-layer impact computation tests.

Focus on correctness of which downstream artifacts are invalidated for
common edit scenarios, and that write / validation errors surfaces clearly.
"""

from __future__ import annotations

import unittest

from feverslop.domain.timeline_editing import (
    BeatMarker,
    EditableTimelineSegment,
    SceneBoundary,
    TimelineSnapshot,
)
from feverslop.ports.timeline_documents import AffectedArtifacts


def _seg(start: float, end: float, **kw) -> EditableTimelineSegment:
    return EditableTimelineSegment(
        start=start, end=end, kind=kw.pop("kind", "vocals"),
        text=kw.pop("text", "test"), **kw,
    )


def _bnd(start: float, end: float, reason: str = "test") -> SceneBoundary:
    return SceneBoundary(start=start, end=end, reason=reason)


def _mk(time_s: float, label: str = "beat") -> BeatMarker:
    return BeatMarker(time_s=time_s, label=label, confidence=0.9)


def _snapshot(**kw) -> TimelineSnapshot:
    return TimelineSnapshot(
        segments=kw.pop("segments", []),
        scene_boundaries=kw.pop("boundaries", []),
        beat_markers=kw.pop("markers", []),
        metadata=kw.pop("meta", {}),
    )


class TimelineImpactScenarios(unittest.TestCase):
    """Impact computation correctness for common edit scenarios."""

    def _impact(self, before: TimelineSnapshot, after: TimelineSnapshot) -> AffectedArtifacts:
        from feverslop.application.edit_audio_timeline import ComputeEditImpact
        return ComputeEditImpact(before, after)

    # ---- Segment trim (split) ----

    def test_segment_trim_invalidates_all_downstream(self):
        before = _snapshot(segments=[_seg(0, 10)])
        after = _snapshot(segments=[_seg(0, 5), _seg(5, 10)])
        impact = self._impact(before, after)
        self.assertTrue(impact.timeline)
        self.assertTrue(impact.stage1_segments)
        self.assertTrue(impact.render_plan)

    def test_segment_merge_invalidates_all_downstream(self):
        before = _snapshot(segments=[_seg(0, 5), _seg(5, 10)])
        after = _snapshot(segments=[_seg(0, 10)])
        impact = self._impact(before, after)
        self.assertTrue(impact.timeline)
        self.assertTrue(impact.stage1_segments)
        self.assertTrue(impact.render_plan)

    # ---- Scene boundary change ----

    def test_boundary_shift_invalidates_scene_srt_and_render_plan(self):
        before = _snapshot(boundaries=[_bnd(0, 5)])
        after = _snapshot(boundaries=[_bnd(0, 3), _bnd(3, 5)])
        impact = self._impact(before, after)
        self.assertTrue(impact.scene_srt)
        self.assertTrue(impact.render_plan)
        self.assertFalse(impact.beat_json)
        self.assertFalse(impact.stage1_segments)

    def test_boundary_add_invalidates_scene_srt(self):
        before = _snapshot(boundaries=[_bnd(0, 10)])
        after = _snapshot(boundaries=[_bnd(0, 5), _bnd(5, 10)])
        impact = self._impact(before, after)
        self.assertTrue(impact.scene_srt)
        self.assertTrue(impact.render_plan)

    # ---- Beat change ----

    def test_beat_add_invalidates_beat_json_only(self):
        before = _snapshot(markers=[_mk(1.0)])
        after = _snapshot(markers=[_mk(1.0), _mk(2.0)])
        impact = self._impact(before, after)
        self.assertTrue(impact.beat_json)
        self.assertFalse(impact.timeline)
        self.assertFalse(impact.stage1_segments)
        self.assertFalse(impact.ltx_prompt)
        self.assertFalse(impact.scene_srt)

    def test_beat_remove_invalidates_beat_json(self):
        before = _snapshot(markers=[_mk(1.0), _mk(2.0)])
        after = _snapshot(markers=[_mk(1.0)])
        impact = self._impact(before, after)
        self.assertTrue(impact.beat_json)

    def test_beat_time_change_invalidates_beat_json(self):
        before = _snapshot(markers=[_mk(1.0)])
        after = _snapshot(markers=[_mk(1.5)])
        impact = self._impact(before, after)
        self.assertTrue(impact.beat_json)

    # ---- Text / lyrics changes ----

    def test_text_change_invalidates_ltx_and_render_plan(self):
        before = _snapshot(segments=[_seg(0, 5, text="hello")])
        after = _snapshot(segments=[_seg(0, 5, text="goodbye")])
        impact = self._impact(before, after)
        self.assertTrue(impact.ltx_prompt)
        self.assertTrue(impact.render_plan)
        self.assertTrue(impact.timeline)

    def test_lyrics_change_invalidates_ltx_only(self):
        before = _snapshot(segments=[_seg(0, 5, lyrics_line="la la")])
        after = _snapshot(segments=[_seg(0, 5, lyrics_line="lu lu")])
        impact = self._impact(before, after)
        self.assertTrue(impact.ltx_prompt)
        self.assertFalse(impact.scene_srt)
        self.assertFalse(impact.beat_json)

    def test_notes_change_only_invalidates_timeline(self):
        before = _snapshot(segments=[_seg(0, 5, notes="fix later")])
        after = _snapshot(segments=[_seg(0, 5, notes="fixed")])
        impact = self._impact(before, after)
        self.assertTrue(impact.timeline)
        self.assertFalse(impact.ltx_prompt)
        self.assertFalse(impact.render_plan)

    # ---- Kind change ----

    def test_kind_change_invalidates_stage1(self):
        before = _snapshot(segments=[_seg(0, 5, kind="vocals")])
        after = _snapshot(segments=[_seg(0, 5, kind="instrumental")])
        impact = self._impact(before, after)
        self.assertTrue(impact.stage1_segments)
        self.assertTrue(impact.timeline)

    # ---- Metadata change ----

    def test_metadata_change_invalidates_render_plan(self):
        before = _snapshot(meta={"project": "a"})
        after = _snapshot(meta={"project": "b"})
        impact = self._impact(before, after)
        self.assertTrue(impact.render_plan)
        self.assertFalse(impact.timeline)

    # ---- No change ----

    def test_identical_snapshots_no_impact(self):
        snap = _snapshot(segments=[_seg(0, 5)], boundaries=[_bnd(0, 5)], markers=[_mk(1.0)])
        impact = self._impact(snap, snap)
        self.assertFalse(impact.any())

    def test_empty_snapshots_no_impact(self):
        before = _snapshot()
        after = _snapshot()
        impact = self._impact(before, after)
        self.assertFalse(impact.any())

    # ---- Composite changes ----

    def test_text_and_boundary_change_invalidates_both(self):
        before = _snapshot(
            segments=[_seg(0, 5, text="hello")],
            boundaries=[_bnd(0, 5)],
        )
        after = _snapshot(
            segments=[_seg(0, 5, text="goodbye")],
            boundaries=[_bnd(0, 3), _bnd(3, 5)],
        )
        impact = self._impact(before, after)
        self.assertTrue(impact.ltx_prompt)
        self.assertTrue(impact.render_plan)
        self.assertTrue(impact.scene_srt)
        self.assertTrue(impact.timeline)


class WriteErrorSurfaceTest(unittest.TestCase):
    """Write errors should not be swallowed by use cases."""

    def test_write_error_surfaces_in_save(self):
        from feverslop.application.edit_audio_timeline import SaveTimelines

        class ErrorReadPort:
            def read_timeline(self): return []
            def read_scene_srt(self): return None
            def read_beat_json(self): return None
            def read_stage1_segments(self): return None
            def read_ltx_prompt_relay(self): return None
            def read_render_plan(self): return None

        class ErrorWritePort:
            def write_timeline(self, data): raise OSError("write failed")
            def write_scene_srt(self, content): pass

        with self.assertRaises(OSError):
            SaveTimelines(ErrorReadPort(), ErrorWritePort()).execute()

    def test_srt_write_error_surfaces_in_save(self):
        from feverslop.application.edit_audio_timeline import SaveTimelines

        class ErrorReadPort:
            def read_timeline(self): return []
            def read_scene_srt(self): return "srt content"
            def read_beat_json(self): return None
            def read_stage1_segments(self): return None
            def read_ltx_prompt_relay(self): return None
            def read_render_plan(self): return None

        class ErrorWritePort:
            def write_timeline(self, data): pass
            def write_scene_srt(self, content): raise OSError("srt write failed")

        with self.assertRaises(OSError):
            SaveTimelines(ErrorReadPort(), ErrorWritePort()).execute()


class ValidationFailureSurfaceTest(unittest.TestCase):
    """Validation failures from domain should surface clearly."""

    def test_negative_start_surfaces_in_edit(self):
        from feverslop.application.edit_audio_timeline import EditAudioTimeline, EditError

        class OkReadPort:
            def read_timeline(self):
                return [
                    {
                        "segments": [{"start": 0, "end": 5, "kind": "vocals", "text": "x",
                                      "lyrics_line": None, "notes": None, "is_draft": False}],
                        "scene_boundaries": [],
                        "beat_markers": [],
                        "metadata": {},
                    }
                ]
            def read_scene_srt(self): return None
            def read_beat_json(self): return None
            def read_stage1_segments(self): return None
            def read_ltx_prompt_relay(self): return None
            def read_render_plan(self): return None

        class OkWritePort:
            written: list = []
            def write_timeline(self, data): self.written.append(data)
            def write_scene_srt(self, content): pass

        uc = EditAudioTimeline(OkReadPort(), OkWritePort())
        with self.assertRaises(EditError) as ctx:
            uc.edit(segment_index=0, changes={"start": -1.0})
        self.assertIn("start", str(ctx.exception))

    def test_end_before_start_surfaces_in_edit(self):
        from feverslop.application.edit_audio_timeline import EditAudioTimeline, EditError

        class OkReadPort:
            def read_timeline(self):
                return [
                    {
                        "segments": [{"start": 0, "end": 5, "kind": "vocals", "text": "x",
                                      "lyrics_line": None, "notes": None, "is_draft": False}],
                        "scene_boundaries": [],
                        "beat_markers": [],
                        "metadata": {},
                    }
                ]
            def read_scene_srt(self): return None
            def read_beat_json(self): return None
            def read_stage1_segments(self): return None
            def read_ltx_prompt_relay(self): return None
            def read_render_plan(self): return None

        class OkWritePort:
            written: list = []
            def write_timeline(self, data): self.written.append(data)
            def write_scene_srt(self, content): pass

        uc = EditAudioTimeline(OkReadPort(), OkWritePort())
        with self.assertRaises(EditError) as ctx:
            uc.edit(segment_index=0, changes={"start": 10.0, "end": 2.0})
        self.assertIn("end", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
