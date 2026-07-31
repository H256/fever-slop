"""Integration tests for the timeline editing stack.

Validates the full flow: domain -> ports -> adapters -> application -> studio
using real file-backed adapters (no mocks for adapter file I/O).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from feverslop.adapters.project_timeline_documents import ProjectTimelineDocuments
from feverslop.application.timeline_app import TimelineAppService
from feverslop.domain.timeline_editing import (
    TimelineSnapshot,
)
from feverslop.ports.timeline_documents import AffectedArtifacts
from feverslop.studio.jobs import JobRegistry
from feverslop.studio.timeline_service import TimelineStudioService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_timeline_json(project_dir: Path, segments: list[dict[str, Any]] | None = None) -> None:
    """Write a minimal timeline.json into project_dir/render/timing/."""
    timeline_path = project_dir / "render/timing/timeline.json"
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    payload = segments if segments is not None else []
    timeline_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_timeline_json(project_dir: Path) -> list[dict[str, Any]]:
    """Read timeline.json and return parsed content."""
    timeline_path = project_dir / "render/timing/timeline.json"
    if not timeline_path.exists():
        return []
    return json.loads(timeline_path.read_text(encoding="utf-8"))


def _default_seed() -> list[dict[str, Any]]:
    """Return a minimal seed timeline with one segment."""
    return [
        {
            "segments": [
                {"start": 0.0, "end": 10.0, "kind": "vocals", "text": "verse one"},
                {"start": 10.0, "end": 20.0, "kind": "vocals", "text": "chorus"},
            ],
            "scene_boundaries": [],
            "beat_markers": [],
            "metadata": {"source": "test"},
        }
    ]


# ---------------------------------------------------------------------------
# Test 1: Full flow — adapter -> app service -> edit -> verify artifacts
# ---------------------------------------------------------------------------


class TestFullEditFlow(unittest.TestCase):
    """End-to-end: seed files, edit via TimelineAppService, verify persistence."""

    def test_edit_segment_persists_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _seed_timeline_json(project_dir, _default_seed())

            adapter = ProjectTimelineDocuments(project_dir)
            service = TimelineAppService(adapter, adapter)

            affected = service.edit_segment(index=0, start_delta=1.0)

            # Verify AffectedArtifacts returned
            self.assertIsInstance(affected, AffectedArtifacts)
            self.assertTrue(affected.timeline)

            # Verify file was updated
            saved = _read_timeline_json(project_dir)
            self.assertEqual(len(saved), 1)
            seg = saved[0]["segments"][0]
            self.assertEqual(seg["start"], 1.0)
            self.assertEqual(seg["end"], 10.0)

    def test_split_segment_persists_two_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _seed_timeline_json(project_dir, _default_seed())

            adapter = ProjectTimelineDocuments(project_dir)
            service = TimelineAppService(adapter, adapter)

            service.split_segment_at(index=0, at=5.0)

            saved = _read_timeline_json(project_dir)
            segs = saved[0]["segments"]
            self.assertEqual(len(segs), 3)  # original 2 + 1 from split
            self.assertEqual(segs[0]["end"], 5.0)
            self.assertEqual(segs[1]["start"], 5.0)

    def test_merge_segments_persists_single_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _seed_timeline_json(project_dir, _default_seed())

            adapter = ProjectTimelineDocuments(project_dir)
            service = TimelineAppService(adapter, adapter)

            affect = service.merge_segments_at(index=0, count=2)

            self.assertTrue(affect.timeline)
            saved = _read_timeline_json(project_dir)
            segs = saved[0]["segments"]
            self.assertEqual(len(segs), 1)
            self.assertEqual(segs[0]["start"], 0.0)
            self.assertEqual(segs[0]["end"], 20.0)
            self.assertEqual(segs[0]["text"], "verse one chorus")

    def test_add_scene_boundary_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _seed_timeline_json(project_dir, _default_seed())

            adapter = ProjectTimelineDocuments(project_dir)
            service = TimelineAppService(adapter, adapter)

            affect = service.add_scene_boundary(0.0, 10.0, "verse")
            self.assertTrue(affect.scene_srt)
            self.assertTrue(affect.render_plan)

            saved = _read_timeline_json(project_dir)
            bnds = saved[0]["scene_boundaries"]
            self.assertEqual(len(bnds), 1)
            self.assertEqual(bnds[0]["reason"], "verse")

    def test_add_beat_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _seed_timeline_json(project_dir, _default_seed())

            adapter = ProjectTimelineDocuments(project_dir)
            service = TimelineAppService(adapter, adapter)

            affect = service.add_beat(1.0, "kick", 0.95)
            self.assertTrue(affect.beat_json)

            saved = _read_timeline_json(project_dir)
            beats = saved[0]["beat_markers"]
            self.assertEqual(len(beats), 1)
            self.assertEqual(beats[0]["time_s"], 1.0)

    def test_multiple_edits_accumulate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _seed_timeline_json(project_dir, _default_seed())

            adapter = ProjectTimelineDocuments(project_dir)
            service = TimelineAppService(adapter, adapter)

            # First edit
            service.edit_segment(0, start_delta=1.0)
            service.edit_segment(1, end_delta=2.0)

            saved = _read_timeline_json(project_dir)
            segs = saved[0]["segments"]
            self.assertEqual(segs[0]["start"], 1.0)
            self.assertEqual(segs[1]["end"], 22.0)


# ---------------------------------------------------------------------------
# Test 2: Studio service undo/redo with persistence
# ---------------------------------------------------------------------------


class TestStudioUndoRedoPersistence(unittest.TestCase):
    """Undo/redo writes to disk; re-loading reads back correct snapshot."""

    def _setup_and_load(
        self,
        project_dir: Path,
        job_registry: JobRegistry,
    ) -> TimelineStudioService:
        service = TimelineStudioService(job_registry=job_registry)
        service.set_project_dir(str(project_dir))
        service.load()
        return service

    def test_edit_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _seed_timeline_json(project_dir, _default_seed())
            registry = JobRegistry()
            service = self._setup_and_load(project_dir, registry)

            service.edit_segment(0, start_delta=5.0)
            saved = _read_timeline_json(project_dir)
            self.assertEqual(saved[0]["segments"][0]["start"], 5.0)

    def test_undo_reverts_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _seed_timeline_json(project_dir, _default_seed())
            registry = JobRegistry()
            service = self._setup_and_load(project_dir, registry)

            service.edit_segment(0, start_delta=5.0)
            service.undo()

            # File should revert to original
            saved = _read_timeline_json(project_dir)
            self.assertEqual(saved[0]["segments"][0]["start"], 0.0)

    def test_redo_reapplies_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _seed_timeline_json(project_dir, _default_seed())
            registry = JobRegistry()
            service = self._setup_and_load(project_dir, registry)

            service.edit_segment(0, start_delta=5.0)
            service.undo()
            service.redo()

            saved = _read_timeline_json(project_dir)
            self.assertEqual(saved[0]["segments"][0]["start"], 5.0)

    def test_multiple_edits_then_full_undo(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            seed = _default_seed()
            _seed_timeline_json(project_dir, seed)
            registry = JobRegistry()
            service = self._setup_and_load(project_dir, registry)

            service.edit_segment(0, start_delta=3.0)
            service.edit_segment(0, start_delta=3.0)
            service.undo()
            service.undo()

            saved = _read_timeline_json(project_dir)
            self.assertEqual(saved[0]["segments"][0]["start"], 0.0)

    def test_reloads_after_undo_see_file_state(self) -> None:
        """After undo, reload() should see the reverted state on disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _seed_timeline_json(project_dir, _default_seed())
            registry = JobRegistry()
            service = self._setup_and_load(project_dir, registry)

            service.edit_segment(0, start_delta=5.0)
            service.undo()

            # Reload and check
            snap = service.load()
            self.assertEqual(snap.segments[0].start, 0.0)


# ---------------------------------------------------------------------------
# Test 3: Error propagation through stack
# ---------------------------------------------------------------------------


class TestErrorPropagation(unittest.TestCase):
    """Domain validation errors propagate through ports -> app -> studio."""

    def test_app_edit_segment_out_of_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _seed_timeline_json(project_dir, _default_seed())

            adapter = ProjectTimelineDocuments(project_dir)
            service = TimelineAppService(adapter, adapter)

            with self.assertRaises(ValueError) as ctx:
                service.edit_segment(index=99, start_delta=1.0)
            self.assertIn("99", str(ctx.exception))

    def test_app_split_segment_out_of_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _seed_timeline_json(project_dir, _default_seed())

            adapter = ProjectTimelineDocuments(project_dir)
            service = TimelineAppService(adapter, adapter)

            with self.assertRaises(ValueError):
                service.split_segment_at(index=99, at=5.0)

    def test_app_merge_too_few_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _seed_timeline_json(project_dir, _default_seed())

            adapter = ProjectTimelineDocuments(project_dir)
            service = TimelineAppService(adapter, adapter)

            with self.assertRaises(ValueError) as ctx:
                service.merge_segments_at(index=0, count=5)
            self.assertIn("out of range", str(ctx.exception))

    def test_app_merge_count_too_small(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _seed_timeline_json(project_dir, _default_seed())

            adapter = ProjectTimelineDocuments(project_dir)
            service = TimelineAppService(adapter, adapter)

            with self.assertRaises(ValueError) as ctx:
                service.merge_segments_at(index=0, count=1)
            self.assertIn("count >= 2", str(ctx.exception))

    def test_app_merge_non_adjacent(self) -> None:
        """Non-adjacent segments trigger domain ValueError through the stack."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            seed: list[dict[str, Any]] = [
                {
                    "segments": [
                        {"start": 0.0, "end": 3.0, "kind": "vocals", "text": "a"},
                        {"start": 5.0, "end": 10.0, "kind": "vocals", "text": "b"},
                    ],
                    "scene_boundaries": [],
                    "beat_markers": [],
                    "metadata": {},
                }
            ]
            _seed_timeline_json(project_dir, seed)

            adapter = ProjectTimelineDocuments(project_dir)
            service = TimelineAppService(adapter, adapter)

            with self.assertRaises(ValueError) as ctx:
                service.merge_segments_at(index=0, count=2)
            self.assertIn("gap", str(ctx.exception).lower())

    def test_app_add_overlapping_boundary(self) -> None:
        """Overlapping scene boundaries trigger domain ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            seed: list[dict[str, Any]] = [
                {
                    "segments": [],
                    "scene_boundaries": [
                        {"start": 0.0, "end": 10.0, "reason": "first", "min_duration": 2.0},
                    ],
                    "beat_markers": [],
                    "metadata": {},
                }
            ]
            _seed_timeline_json(project_dir, seed)

            adapter = ProjectTimelineDocuments(project_dir)
            service = TimelineAppService(adapter, adapter)

            with self.assertRaises(ValueError) as ctx:
                service.add_scene_boundary(5.0, 12.0, "overlap")
            self.assertIn("overlap", str(ctx.exception).lower())

    def test_app_add_too_short_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _seed_timeline_json(project_dir, _default_seed())

            adapter = ProjectTimelineDocuments(project_dir)
            service = TimelineAppService(adapter, adapter)

            with self.assertRaises(ValueError):
                service.add_scene_boundary(0.0, 1.0, "too short")

    def test_app_add_duplicate_beat(self) -> None:
        """Duplicate beat times trigger domain ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            seed: list[dict[str, Any]] = [
                {
                    "segments": [],
                    "scene_boundaries": [],
                    "beat_markers": [
                        {"time_s": 1.0, "label": "existing", "confidence": 0.9},
                    ],
                    "metadata": {},
                }
            ]
            _seed_timeline_json(project_dir, seed)

            adapter = ProjectTimelineDocuments(project_dir)
            service = TimelineAppService(adapter, adapter)

            with self.assertRaises(ValueError) as ctx:
                service.add_beat(1.0, "dup", 0.9)
            self.assertIn("duplicate", str(ctx.exception).lower())

    def test_app_add_invalid_confidence_beat(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _seed_timeline_json(project_dir, _default_seed())

            adapter = ProjectTimelineDocuments(project_dir)
            service = TimelineAppService(adapter, adapter)

            with self.assertRaises(ValueError):
                service.add_beat(1.0, "bad", 2.0)

    def test_studio_edit_segment_negative_start(self) -> None:
        """Editing a segment to have negative start triggers domain ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _seed_timeline_json(project_dir, _default_seed())
            registry = JobRegistry()
            service = TimelineStudioService(job_registry=registry)
            service.set_project_dir(str(project_dir))
            service.load()

            with self.assertRaises(ValueError):
                service.edit_segment(0, start_delta=-5.0)

    def test_studio_error_without_project(self) -> None:
        """Studio operations without set_project_dir raise RuntimeError."""
        registry = JobRegistry()
        service = TimelineStudioService(job_registry=registry)

        with self.assertRaises(RuntimeError):
            service.edit_segment(0)

    def test_studio_error_without_load(self) -> None:
        """Studio operations without load() raise RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _seed_timeline_json(project_dir, _default_seed())
            registry = JobRegistry()
            service = TimelineStudioService(job_registry=registry)
            service.set_project_dir(str(project_dir))

            with self.assertRaises(RuntimeError):
                service.edit_segment(0, start_delta=1.0)


# ---------------------------------------------------------------------------
# Test 4: Pipeline rebuild job scheduling
# ---------------------------------------------------------------------------


class TestRebuildJobFlow(unittest.TestCase):
    """Rebuild pipeline correctly schedules jobs via JobRegistry."""

    def test_studio_rebuild_pipeline_registers_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _seed_timeline_json(project_dir, _default_seed())
            registry = JobRegistry()
            service = TimelineStudioService(job_registry=registry)
            service.set_project_dir(str(project_dir))
            service.load()

            # Perform an edit first (creates AffectedArtifacts)
            affected = service.edit_segment(0, start_delta=1.0)
            job = service.rebuild_pipeline(affected)

            self.assertIsInstance(job, dict)
            self.assertIn("id", job)
            self.assertEqual(job["action"], "rebuild-plan-timeline")
            self.assertIn("rebuild_id", job)

    def test_rebuild_pipeline_job_in_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _seed_timeline_json(project_dir, _default_seed())
            registry = JobRegistry()
            service = TimelineStudioService(job_registry=registry)
            service.set_project_dir(str(project_dir))
            service.load()

            affected = service.edit_segment(0, start_delta=1.0)
            service.rebuild_pipeline(affected)

            import time
            time.sleep(0.05)  # Let thread start
            jobs = registry.list()
            rebuild_jobs = [j for j in jobs if j.get("action") == "rebuild-plan-timeline"]
            self.assertTrue(len(rebuild_jobs) >= 1)

    def test_rebuild_pipeline_propagates_affected_flags(self) -> None:
        """The scheduled job payload contains the correct affected flags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _seed_timeline_json(project_dir, _default_seed())
            registry = JobRegistry()
            service = TimelineStudioService(job_registry=registry)
            service.set_project_dir(str(project_dir))
            service.load()

            affected = service.edit_segment(0, start_delta=1.0)
            affected.beat_json = True
            affected.scene_srt = True
            job = service.rebuild_pipeline(affected)

            payload = job.get("payload", {})
            self.assertTrue(payload["affected"]["beat_json"])
            self.assertTrue(payload["affected"]["scene_srt"])
            self.assertEqual(payload["project_dir"], str(project_dir.resolve()))

    def test_rebuild_pipeline_requires_loaded_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            _seed_timeline_json(project_dir, _default_seed())
            registry = JobRegistry()
            service = TimelineStudioService(job_registry=registry)
            service.set_project_dir(str(project_dir))

            with self.assertRaises(RuntimeError):
                service.rebuild_pipeline(AffectedArtifacts())


# ---------------------------------------------------------------------------
# Test 5: Empty project loading
# ---------------------------------------------------------------------------


class TestEmptyProjectFlow(unittest.TestCase):
    """Loading and editing projects with no timeline files."""

    def test_adapter_returns_empty_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            adapter = ProjectTimelineDocuments(project_dir)
            result = adapter.read_timeline()
            self.assertEqual(result, [])

    def test_app_service_fails_on_edits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            # No timeline.json exists
            adapter = ProjectTimelineDocuments(project_dir)
            service = TimelineAppService(adapter, adapter)

            with self.assertRaises(ValueError) as ctx:
                service.edit_segment(index=0, start_delta=1.0)
            self.assertIn("out of range", str(ctx.exception))

    def test_studio_service_creates_empty_snapshot(self) -> None:
        """Studio service load() on empty project yields empty snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            registry = JobRegistry()
            service = TimelineStudioService(job_registry=registry)
            service.set_project_dir(str(project_dir))
            snap = service.load()

            self.assertIsInstance(snap, TimelineSnapshot)
            self.assertEqual(len(snap.segments), 0)
            self.assertEqual(len(snap.scene_boundaries), 0)
            self.assertEqual(len(snap.beat_markers), 0)

    def test_studio_service_save_empty_then_reload(self) -> None:
        """Save an empty timeline, then reload to see empty data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            registry = JobRegistry()
            service = TimelineStudioService(job_registry=registry)
            service.set_project_dir(str(project_dir))
            service.load()  # empty snapshot
            service.save()  # writes empty

            # Verify file exists and contains valid empty data
            saved = _read_timeline_json(project_dir)
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["segments"], [])

            # Reload should also be empty
            snap2 = service.load()
            self.assertEqual(len(snap2.segments), 0)

    def test_studio_add_boundary_to_empty_timeline(self) -> None:
        """Can add scene boundaries to an empty timeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            registry = JobRegistry()
            service = TimelineStudioService(job_registry=registry)
            service.set_project_dir(str(project_dir))
            service.load()

            affect = service.add_scene_boundary(0.0, 5.0, "opening")
            self.assertTrue(affect.scene_srt)

            snap = service.current_snapshot()
            self.assertEqual(len(snap.scene_boundaries), 1)
            self.assertEqual(len(snap.segments), 0)

    def test_studio_add_beat_to_empty_timeline(self) -> None:
        """Can add beat markers to an empty timeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            registry = JobRegistry()
            service = TimelineStudioService(job_registry=registry)
            service.set_project_dir(str(project_dir))
            service.load()

            affect = service.add_beat(0.5, "start", 1.0)
            self.assertTrue(affect.beat_json)

            snap = service.current_snapshot()
            self.assertEqual(len(snap.beat_markers), 1)

    def test_studio_edit_fail_on_empty_timeline(self) -> None:
        """Editing a non-existent segment on empty timeline raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            registry = JobRegistry()
            service = TimelineStudioService(job_registry=registry)
            service.set_project_dir(str(project_dir))
            service.load()

            with self.assertRaises(ValueError):
                service.edit_segment(0, start_delta=1.0)


if __name__ == "__main__":
    unittest.main()
