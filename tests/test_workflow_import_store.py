"""Focused tests for the per-project workflow import store (P3)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from feverslop.domain.workflow_import import TestRunResult
from feverslop.domain.workflow_import_inspector import inspect_workflow, validate_workflow
from feverslop.domain.workflow_import_families import spec_for
from feverslop.domain.workflow_import_store import (
    WorkflowImportError,
    WorkflowImportStore,
)

_GOOD_GRAPH = {
    "1": {"class_type": "LoadImage", "inputs": {}},
    "2": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {"audio": [3, 0]}},
    "3": {"class_type": "LoadAudio", "inputs": {}},
    "4": {"class_type": "VAEDecode", "inputs": {"samples": [2, 0]}},
}


class StoreLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = WorkflowImportStore(projects_root=Path(self._tmp.name))

    def test_import_creates_draft_with_sha_snapshot(self) -> None:
        profile = self.store.import_workflow(
            project_id="proj", profile_id="h3-final", pipeline="minimax_h3",
            purpose="final", graph=_GOOD_GRAPH,
        )
        self.assertEqual("draft", profile.status)
        self.assertEqual(len(profile.workflow_sha256), 64)
        self.assertTrue(self.store.verify_snapshot("proj", "h3-final"))
        self.assertFalse(self.store.is_snapshot_pinned("proj", "h3-final"))

    def test_import_rejects_bad_profile_id_and_purpose(self) -> None:
        with self.assertRaises(WorkflowImportError):
            self.store.import_workflow(
                project_id="p", profile_id="Bad ID", pipeline="x", purpose="final",
                graph=_GOOD_GRAPH,
            )
        with self.assertRaises(WorkflowImportError):
            self.store.import_workflow(
                project_id="p", profile_id="ok", pipeline="x", purpose="weird",
                graph=_GOOD_GRAPH,
            )

    def test_activation_requires_validation_and_test_run(self) -> None:
        self.store.import_workflow(
            project_id="p", profile_id="h3", pipeline="minimax_h3", purpose="final",
            graph=_GOOD_GRAPH,
        )
        with self.assertRaises(WorkflowImportError):
            self.store.activate("p", "h3")  # no validation yet
        self.store.record_validation("p", "h3", valid=True)
        with self.assertRaises(WorkflowImportError):
            self.store.activate("p", "h3")  # no test-run yet
        self.store.record_test_run("p", "h3", TestRunResult(success=True, detail="ok"))
        self.store.activate("p", "h3")
        self.assertEqual("active", self.store.get_profile("p", "h3").status)

    def test_failed_validation_or_test_run_marks_broken(self) -> None:
        self.store.import_workflow(
            project_id="p", profile_id="h3", pipeline="minimax_h3", purpose="final",
            graph=_GOOD_GRAPH,
        )
        self.store.record_validation("p", "h3", valid=False)
        self.assertEqual("broken", self.store.get_profile("p", "h3").status)
        self.store.set_state("p", "h3", "draft")
        self.store.record_validation("p", "h3", valid=True)
        self.store.record_test_run("p", "h3", TestRunResult(success=False, detail="boom"))
        self.assertEqual("broken", self.store.get_profile("p", "h3").status)

    def test_illegal_state_transition_rejected(self) -> None:
        self.store.import_workflow(
            project_id="p", profile_id="h3", pipeline="minimax_h3", purpose="final",
            graph=_GOOD_GRAPH,
        )
        with self.assertRaises(WorkflowImportError):
            self.store.set_state("p", "h3", "active")  # draft cannot jump to active

    def test_deactivate_only_from_active(self) -> None:
        self.store.import_workflow(
            project_id="p", profile_id="h3", pipeline="minimax_h3", purpose="final",
            graph=_GOOD_GRAPH,
        )
        with self.assertRaises(WorkflowImportError):
            self.store.deactivate("p", "h3")
        self.store.record_validation("p", "h3", valid=True)
        self.store.record_test_run("p", "h3", TestRunResult(success=True))
        self.store.activate("p", "h3")
        self.store.deactivate("p", "h3")
        self.assertEqual("tested", self.store.get_profile("p", "h3").status)

    def test_pins_track_render_ids(self) -> None:
        self.store.import_workflow(
            project_id="p", profile_id="h3", pipeline="minimax_h3", purpose="final",
            graph=_GOOD_GRAPH,
        )
        self.store.pin_snapshot("p", "h3", "render-1")
        self.store.pin_snapshot("p", "h3", "render-1")  # idempotent
        self.assertTrue(self.store.is_snapshot_pinned("p", "h3"))
        self.store.unpin_snapshot("p", "h3", "render-1")
        self.assertFalse(self.store.is_snapshot_pinned("p", "h3"))

    def test_list_profiles_and_roundtrip(self) -> None:
        self.store.import_workflow(
            project_id="p", profile_id="a", pipeline="minimax_h3", purpose="final",
            graph=_GOOD_GRAPH,
        )
        self.store.import_workflow(
            project_id="p", profile_id="b", pipeline="minimax_h3", purpose="preview",
            graph=_GOOD_GRAPH,
        )
        profiles = self.store.list_profiles("p")
        self.assertEqual(["a", "b"], [p.profile_id for p in profiles])
        self.assertEqual("final", self.store.get_profile("p", "a").purpose)

    def test_snapshot_integrity_detection(self) -> None:
        self.store.import_workflow(
            project_id="p", profile_id="h3", pipeline="minimax_h3", purpose="final",
            graph=_GOOD_GRAPH,
        )
        self.assertTrue(self.store.verify_snapshot("p", "h3"))
        snapshot = (
            Path(self._tmp.name) / "p" / "workflows" / "h3"
            / f"{self.store.get_profile('p', 'h3').workflow_sha256[:16]}.json"
        )
        snapshot.write_bytes(b"tampered")
        self.assertFalse(self.store.verify_snapshot("p", "h3"))

    def test_reimport_self_heals_corrupted_snapshot(self) -> None:
        self.store.import_workflow(
            project_id="p", profile_id="h3", pipeline="minimax_h3", purpose="final",
            graph=_GOOD_GRAPH,
        )
        snapshot = (
            Path(self._tmp.name) / "p" / "workflows" / "h3"
            / f"{self.store.get_profile('p', 'h3').workflow_sha256[:16]}.json"
        )
        # Simulate an interrupted write leaving a truncated payload.
        snapshot.write_bytes(snapshot.read_bytes()[:5])
        self.assertFalse(self.store.verify_snapshot("p", "h3"))
        # Re-importing the same graph must restore a verifiable snapshot.
        self.store.import_workflow(
            project_id="p", profile_id="h3", pipeline="minimax_h3", purpose="final",
            graph=_GOOD_GRAPH,
        )
        self.assertTrue(self.store.verify_snapshot("p", "h3"))
        self.assertEqual(
            self.store.snapshot_bytes("p", "h3"),
            snapshot.read_bytes(),
        )

    def test_snapshot_write_is_atomic(self) -> None:
        self.store.import_workflow(
            project_id="p", profile_id="h3", pipeline="minimax_h3", purpose="final",
            graph=_GOOD_GRAPH,
        )
        # No stray temp files may be left behind by the atomic write.
        directory = Path(self._tmp.name) / "p" / "workflows" / "h3"
        leftovers = [p.name for p in directory.iterdir() if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [])


class StoreWithInspectorTests(unittest.TestCase):
    def test_validation_records_mapping_and_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = WorkflowImportStore(projects_root=Path(tmp))
            store.import_workflow(
                project_id="p", profile_id="h3", pipeline="minimax_h3", purpose="final",
                graph=_GOOD_GRAPH,
            )
            spec = spec_for("minimax_h3")
            assert spec is not None
            report = validate_workflow(_GOOD_GRAPH, spec=spec)
            analysis = inspect_workflow(_GOOD_GRAPH, spec=spec)
            store.record_validation("p", "h3", valid=report.valid, analysis=analysis)
            profile = store.get_profile("p", "h3")
            self.assertEqual("validated", profile.status)
            assert profile.mapping is not None
            self.assertEqual("2", profile.mapping.core_node_id)
            assert profile.analysis is not None
            self.assertEqual("auto_compatible", profile.analysis.compatibility)


if __name__ == "__main__":
    unittest.main()
