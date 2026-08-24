from __future__ import annotations

import copy
import unittest
from dataclasses import FrozenInstanceError
from typing import Any

from feverslop.domain.timeline_editing import (
    TimelineEditImpact,
)
from feverslop.ports.timeline_documents import (
    AffectedArtifacts,
    TimelineReadPort,
    TimelineWritePort,
)

# ---------------------------------------------------------------------------
# Fake port implementations for testing
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
        self.write_errors: list[str] = []

    def write_timeline(self, data: list[dict]) -> None:
        if self.raise_on_write:
            raise OSError("disk full")
        self.timeline = copy.deepcopy(data)

    def write_scene_srt(self, content: str) -> None:
        if self.raise_on_write:
            raise OSError("disk full")
        self.scene_srt = content


class ErrorsOnWritePort:
    """Write port that always raises, for error-propagation tests."""

    def write_timeline(self, data: list[dict]) -> None:
        raise OSError("simulated write failure")

    def write_scene_srt(self, content: str) -> None:
        raise OSError("simulated write failure")


# ---------------------------------------------------------------------------
# Port interface tests (Protocol conformance)
# ---------------------------------------------------------------------------


class TimelineReadPortInterfaceTest(unittest.TestCase):
    """Verify FakeReadPort satisfies the TimelineReadPort protocol."""

    def test_read_timeline_returns_copy(self):
        port: TimelineReadPort = FakeReadPort()
        port.timeline = [{"start": 0.0, "end": 5.0}]
        data = port.read_timeline()
        self.assertEqual(len(data), 1)
        # Mutation of returned data should not affect internal store.
        data[0]["start"] = 999.0
        self.assertEqual(port.timeline[0]["start"], 0.0)

    def test_read_scene_srt_returns_none_when_absent(self):
        port: TimelineReadPort = FakeReadPort()
        self.assertIsNone(port.read_scene_srt())

    def test_read_scene_srt_returns_content(self):
        port: TimelineReadPort = FakeReadPort()
        port.scene_srt = "1\n0:00:00,000 --> 0:00:05,000\nVerse"
        self.assertEqual(port.read_scene_srt(), "1\n0:00:00,000 --> 0:00:05,000\nVerse")

    def test_read_beat_json_returns_none_when_absent(self):
        port: TimelineReadPort = FakeReadPort()
        self.assertIsNone(port.read_beat_json())

    def test_read_beat_json_returns_deepcopy(self):
        port: TimelineReadPort = FakeReadPort()
        port.beat_json = [{"time_s": 1.0}]
        data = port.read_beat_json()
        self.assertEqual(len(data), 1)
        data[0]["time_s"] = 999.0
        self.assertEqual(port.beat_json[0]["time_s"], 1.0)

    def test_read_stage1_segments_returns_none_when_absent(self):
        port: TimelineReadPort = FakeReadPort()
        self.assertIsNone(port.read_stage1_segments())

    def test_read_ltx_prompt_relay_returns_none_when_absent(self):
        port: TimelineReadPort = FakeReadPort()
        self.assertIsNone(port.read_ltx_prompt_relay())

    def test_read_render_plan_returns_none_when_absent(self):
        port: TimelineReadPort = FakeReadPort()
        self.assertIsNone(port.read_render_plan())


class TimelineWritePortInterfaceTest(unittest.TestCase):
    """Verify FakeWritePort satisfies the TimelineWritePort protocol."""

    def test_write_timeline_stores_data(self):
        port: TimelineWritePort = FakeWritePort()
        data = [{"start": 0.0, "end": 5.0}]
        port.write_timeline(data)
        self.assertEqual(port.timeline, [{"start": 0.0, "end": 5.0}])

    def test_write_scene_srt_stores_content(self):
        port: TimelineWritePort = FakeWritePort()
        content = "scene srt content"
        port.write_scene_srt(content)
        self.assertEqual(port.scene_srt, content)

    def test_write_timeline_stores_deepcopy(self):
        port: TimelineWritePort = FakeWritePort()
        data = [{"start": 0.0}]
        port.write_timeline(data)
        data[0]["start"] = 999.0
        self.assertEqual(port.timeline[0]["start"], 0.0)


# ---------------------------------------------------------------------------
# AffectedArtifacts tests
# ---------------------------------------------------------------------------


class AffectedArtifactsTest(unittest.TestCase):
    def test_creation_with_defaults(self):
        aa = AffectedArtifacts()
        self.assertFalse(aa.timeline)
        self.assertFalse(aa.scene_srt)
        self.assertFalse(aa.beat_json)
        self.assertFalse(aa.stage1_segments)
        self.assertFalse(aa.ltx_prompt)
        self.assertFalse(aa.render_plan)

    def test_creation_with_flags(self):
        aa = AffectedArtifacts(timeline=True, scene_srt=True)
        self.assertTrue(aa.timeline)
        self.assertTrue(aa.scene_srt)
        self.assertFalse(aa.beat_json)

    def test_from_timeline_edit_impact(self):
        impact = TimelineEditImpact(
            timeline_invalidated=True,
            scene_srt_invalidated=False,
            beat_json_invalidated=True,
            stage1_segments_invalidated=True,
            ltx_prompt_invalidated=False,
            render_plan_invalidated=False,
        )
        aa = AffectedArtifacts.from_timeline_edit_impact(impact)
        self.assertTrue(aa.timeline)
        self.assertFalse(aa.scene_srt)
        self.assertTrue(aa.beat_json)
        self.assertTrue(aa.stage1_segments)
        self.assertFalse(aa.ltx_prompt)
        self.assertFalse(aa.render_plan)

    def test_any_invalidated(self):
        aa_none = AffectedArtifacts()
        self.assertFalse(aa_none.any())
        aa_some = AffectedArtifacts(timeline=True)
        self.assertTrue(aa_some.any())

    def test_all_flags(self):
        aa = AffectedArtifacts(
            timeline=True,
            scene_srt=True,
            beat_json=True,
            stage1_segments=True,
            ltx_prompt=True,
            render_plan=True,
        )
        self.assertTrue(aa.any())

    def test_flags_are_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            AffectedArtifacts(timeline=True).timeline = False


if __name__ == "__main__":
    unittest.main()
