from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from feverslop.adapters.project_timeline_documents import ProjectTimelineDocuments
from feverslop.ports.timeline_documents import TimelineReadPort, TimelineWritePort

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_json(base: Path, rel: str, data) -> Path:
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_text(base: Path, rel: str, text: str) -> Path:
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestProjectTimelineDocumentsReads(unittest.TestCase):
    """Read port methods on ProjectTimelineDocuments."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project_dir = Path(self.tmp.name)

    # -- read_timeline --

    def test_read_timeline_returns_deepcopy(self):
        timeline = [{"start": 0.0, "end": 4.0, "kind": "vocals"}, {"start": 4.0, "end": 8.0, "kind": "instrumental"}]
        _write_json(self.project_dir, "render/timing/timeline.json", timeline)
        adapter = ProjectTimelineDocuments(self.project_dir)
        result = adapter.read_timeline()
        self.assertEqual(result, timeline)
        # Mutation of result must not affect the file.
        result[0]["start"] = 999.0
        second = adapter.read_timeline()
        self.assertEqual(second[0]["start"], 0.0)

    def test_read_timeline_missing_returns_empty_list(self):
        adapter = ProjectTimelineDocuments(self.project_dir)
        self.assertEqual(adapter.read_timeline(), [])

    # -- read_scene_srt --

    def test_read_scene_srt_returns_content(self):
        srt = "1\n0:00:00,000 --> 0:00:04,000\nVerse one"
        _write_text(self.project_dir, "render/timing/scene_srt", srt)
        adapter = ProjectTimelineDocuments(self.project_dir)
        self.assertEqual(adapter.read_scene_srt(), srt)

    def test_read_scene_srt_missing_returns_none(self):
        adapter = ProjectTimelineDocuments(self.project_dir)
        self.assertIsNone(adapter.read_scene_srt())

    # -- read_beat_json --

    def test_read_beat_json_returns_deepcopy(self):
        beats = [{"time_s": 0.5, "label": "downbeat", "confidence": 0.95}]
        _write_json(self.project_dir, "render/timing/beat_json", beats)
        adapter = ProjectTimelineDocuments(self.project_dir)
        result = adapter.read_beat_json()
        self.assertEqual(result, beats)
        result[0]["time_s"] = 777.0
        second = adapter.read_beat_json()
        self.assertEqual(second[0]["time_s"], 0.5)

    def test_read_beat_json_missing_returns_none(self):
        adapter = ProjectTimelineDocuments(self.project_dir)
        self.assertIsNone(adapter.read_beat_json())

    # -- read_stage1_segments --

    def test_read_stage1_segments_returns_deepcopy(self):
        segments = [{"scene": 1, "start_s": 0.0, "end_s": 4.0}]
        _write_json(self.project_dir, "render/timing/stage1_segments.json", segments)
        adapter = ProjectTimelineDocuments(self.project_dir)
        result = adapter.read_stage1_segments()
        self.assertEqual(result, segments)
        result[0]["scene"] = 99
        second = adapter.read_stage1_segments()
        self.assertEqual(second[0]["scene"], 1)

    def test_read_stage1_segments_missing_returns_none(self):
        adapter = ProjectTimelineDocuments(self.project_dir)
        self.assertIsNone(adapter.read_stage1_segments())

    # -- read_ltx_prompt_relay --

    def test_read_ltx_prompt_relay_returns_deepcopy(self):
        relay = [{"scene": 1, "prompt": "wide shot"}]
        _write_json(self.project_dir, "render/stage1/ltx_prompt_relay.json", relay)
        adapter = ProjectTimelineDocuments(self.project_dir)
        result = adapter.read_ltx_prompt_relay()
        self.assertEqual(result, relay)
        result[0]["prompt"] = "mutated"
        second = adapter.read_ltx_prompt_relay()
        self.assertEqual(second[0]["prompt"], "wide shot")

    def test_read_ltx_prompt_relay_missing_returns_none(self):
        adapter = ProjectTimelineDocuments(self.project_dir)
        self.assertIsNone(adapter.read_ltx_prompt_relay())

    # -- read_render_plan --

    def test_read_render_plan_returns_deepcopy(self):
        plan = {"scenes": 4, "prompt": "cinematic"}
        _write_json(self.project_dir, "render/project_render_plan.json", plan)
        adapter = ProjectTimelineDocuments(self.project_dir)
        result = adapter.read_render_plan()
        self.assertEqual(result, plan)
        result["scenes"] = 99
        second = adapter.read_render_plan()
        self.assertEqual(second["scenes"], 4)

    def test_read_render_plan_missing_returns_none(self):
        adapter = ProjectTimelineDocuments(self.project_dir)
        self.assertIsNone(adapter.read_render_plan())


class TestProjectTimelineDocumentsWrites(unittest.TestCase):
    """Write port methods on ProjectTimelineDocuments."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project_dir = Path(self.tmp.name)

    def test_write_timeline_creates_file(self):
        data = [{"start": 0.0, "end": 5.0, "kind": "vocals"}]
        adapter = ProjectTimelineDocuments(self.project_dir)
        adapter.write_timeline(data)
        target = self.project_dir / "render/timing/timeline.json"
        self.assertTrue(target.exists())
        self.assertEqual(json.loads(target.read_text("utf-8")), data)

    def test_write_timeline_creates_parent_dirs(self):
        data = [{"start": 0.0, "end": 5.0}]
        adapter = ProjectTimelineDocuments(self.project_dir)
        adapter.write_timeline(data)
        target = self.project_dir / "render/timing/timeline.json"
        self.assertTrue(target.exists())
        self.assertTrue(target.parent.is_dir())

    def test_write_scene_srt_creates_file(self):
        content = "1\n0:00:00,000 --> 0:00:05,000\nTest"
        adapter = ProjectTimelineDocuments(self.project_dir)
        adapter.write_scene_srt(content)
        target = self.project_dir / "render/timing/scene_srt"
        self.assertTrue(target.exists())
        self.assertEqual(target.read_text("utf-8"), content)

    def test_write_scene_srt_creates_parent_dirs(self):
        adapter = ProjectTimelineDocuments(self.project_dir)
        adapter.write_scene_srt("content")
        self.assertTrue((self.project_dir / "render/timing").is_dir())


class TestProjectTimelineDocumentsRoundtrip(unittest.TestCase):
    """Read/write roundtrip preserves data."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project_dir = Path(self.tmp.name)

    def test_timeline_roundtrip(self):
        original = [
            {"start": 0.0, "end": 4.0, "kind": "vocals", "text": "Hello"},
            {"start": 4.0, "end": 8.0, "kind": "instrumental"},
        ]
        adapter = ProjectTimelineDocuments(self.project_dir)
        adapter.write_timeline(original)
        loaded = adapter.read_timeline()
        self.assertEqual(loaded, original)

    def test_scene_srt_roundtrip(self):
        original = "1\n0:00:00,000 --> 0:00:05,000\nScene one\n\n2\n0:00:05,000 --> 0:00:10,000\nScene two"
        adapter = ProjectTimelineDocuments(self.project_dir)
        adapter.write_scene_srt(original)
        loaded = adapter.read_scene_srt()
        self.assertEqual(loaded, original)


class TestSnapshotSafety(unittest.TestCase):
    """Verify all reads return snapshot-safe copies."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project_dir = Path(self.tmp.name)

    def test_timeline_mutation_does_not_persist(self):
        original = [{"start": 0.0, "end": 4.0, "kind": "vocals"}]
        _write_json(self.project_dir, "render/timing/timeline.json", original)
        adapter = ProjectTimelineDocuments(self.project_dir)
        first = adapter.read_timeline()
        first.append({"start": 10.0, "end": 20.0, "kind": "extra"})
        second = adapter.read_timeline()
        self.assertEqual(len(second), 1)

    def test_beat_json_mutation_does_not_persist(self):
        original = [{"time_s": 1.0, "label": "beat"}]
        _write_json(self.project_dir, "render/timing/beat_json", original)
        adapter = ProjectTimelineDocuments(self.project_dir)
        first = adapter.read_beat_json()
        assert first is not None
        first[0]["time_s"] = 99.0
        second = adapter.read_beat_json()
        self.assertEqual(second[0]["time_s"], 1.0)

    def test_stage1_segments_mutation_does_not_persist(self):
        original = [{"scene": 1, "start_s": 0.0}]
        _write_json(self.project_dir, "render/timing/stage1_segments.json", original)
        adapter = ProjectTimelineDocuments(self.project_dir)
        first = adapter.read_stage1_segments()
        assert first is not None
        first.clear()
        second = adapter.read_stage1_segments()
        self.assertEqual(len(second), 1)

    def test_ltx_prompt_relay_mutation_does_not_persist(self):
        original = [{"scene": 1, "prompt": "test"}]
        _write_json(self.project_dir, "render/stage1/ltx_prompt_relay.json", original)
        adapter = ProjectTimelineDocuments(self.project_dir)
        first = adapter.read_ltx_prompt_relay()
        assert first is not None
        first[0]["prompt"] = "mutated"
        second = adapter.read_ltx_prompt_relay()
        self.assertEqual(second[0]["prompt"], "test")

    def test_render_plan_mutation_does_not_persist(self):
        original = {"version": 1, "scenes": 3}
        _write_json(self.project_dir, "render/project_render_plan.json", original)
        adapter = ProjectTimelineDocuments(self.project_dir)
        first = adapter.read_render_plan()
        assert first is not None
        first.update({"version": 99})
        second = adapter.read_render_plan()
        self.assertEqual(second["version"], 1)


class TestAllArtifactPaths(unittest.TestCase):
    """Verify every expected artifact path is exercised."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project_dir = Path(self.tmp.name)

    def test_all_read_paths_resolved_correctly(self):
        """Write a sentinel to every read path and verify it is returned."""
        sentinels = {
            "render/timing/timeline.json": [{"sentinel": "timeline"}],
            "render/timing/scene_srt": "scene_srt_sentinel",
            "render/timing/beat_json": [{"sentinel": "beat"}],
            "render/timing/stage1_segments.json": [{"sentinel": "stage1"}],
            "render/stage1/ltx_prompt_relay.json": [{"sentinel": "relay"}],
            "render/project_render_plan.json": {"sentinel": "plan"},
        }
        for rel, value in sentinels.items():
            if isinstance(value, str):
                _write_text(self.project_dir, rel, value)
            else:
                _write_json(self.project_dir, rel, value)

        adapter = ProjectTimelineDocuments(self.project_dir)

        timeline = adapter.read_timeline()
        self.assertEqual(timeline, sentinels["render/timing/timeline.json"])

        srt = adapter.read_scene_srt()
        self.assertEqual(srt, sentinels["render/timing/scene_srt"])

        beats = adapter.read_beat_json()
        self.assertEqual(beats, sentinels["render/timing/beat_json"])

        segments = adapter.read_stage1_segments()
        self.assertEqual(segments, sentinels["render/timing/stage1_segments.json"])

        relay = adapter.read_ltx_prompt_relay()
        self.assertEqual(relay, sentinels["render/stage1/ltx_prompt_relay.json"])

        plan = adapter.read_render_plan()
        self.assertEqual(plan, sentinels["render/project_render_plan.json"])


class TestProtocolConformance(unittest.TestCase):
    """Verify ProjectTimelineDocuments satisfies port protocols."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project_dir = Path(self.tmp.name)

    def test_satisfies_read_port(self):
        # Just verify the methods exist and are callable, returning the
        # expected types (None is fine for missing files).
        adapter: TimelineReadPort = ProjectTimelineDocuments(self.project_dir)
        self.assertIsInstance(adapter.read_timeline(), (list, type(None)))
        self.assertIsInstance(adapter.read_scene_srt(), (str, type(None)))
        self.assertIsInstance(adapter.read_beat_json(), (list, type(None)))
        self.assertIsInstance(adapter.read_stage1_segments(), (list, type(None)))
        self.assertIsInstance(adapter.read_ltx_prompt_relay(), (list, type(None)))
        self.assertIsInstance(adapter.read_render_plan(), (dict, type(None)))

    def test_satisfies_write_port(self):
        adapter: TimelineWritePort = ProjectTimelineDocuments(self.project_dir)
        adapter.write_timeline([{"start": 0, "end": 1}])
        adapter.write_scene_srt("content")


if __name__ == "__main__":
    unittest.main()
