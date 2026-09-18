"""Focused tests for P4 precedence: active import overrides config default."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from feverslop.config.app_config import AppConfig
from feverslop.domain.video_workflow_profile import VideoWorkflowProfile
from feverslop.domain.workflow_import import TestRunResult
from feverslop.domain.workflow_import_store import WorkflowImportStore

_GRAPH = {
    "1": {"class_type": "LoadImage", "inputs": {}},
    "2": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {"audio": [3, 0]}},
    "3": {"class_type": "LoadAudio", "inputs": {}},
    "4": {"class_type": "VAEDecode", "inputs": {"samples": [2, 0]}},
}


def _make_config(tmp: str) -> AppConfig:
    default = VideoWorkflowProfile.create(
        name="h3-default",
        pipeline="minimax_h3",
        workflow_path="workflows/h3_default.json",
        purpose="final",
        stages=1,
        output_scale=1.0,
        supports_per_pass_loras=False,
    )
    return AppConfig(
        llm=None,  # type: ignore[arg-type]
        comfyui=None,  # type: ignore[arg-type]
        video_workflow_profiles=(default,),
        _video_workflow_profile_defaults=(("minimax_h3", "final", "h3-default"),),
    )


class PrecedenceTests(unittest.TestCase):
    def test_no_import_returns_config_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(tmp)
            profile = config.resolve_video_workflow_profile(pipeline="minimax_h3", purpose="final")
            assert profile is not None
            self.assertEqual("workflows/h3_default.json", profile.workflow_path)

    def test_active_import_overrides_config_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = WorkflowImportStore(projects_root=root)
            store.import_workflow(
                project_id="proj", profile_id="h3-final", pipeline="minimax_h3",
                purpose="final", graph=_GRAPH,
            )
            store.record_validation("proj", "h3-final", valid=True)
            store.record_test_run("proj", "h3-final", TestRunResult(success=True))
            store.activate("proj", "h3-final")

            config = _make_config(tmp)
            config.import_store = store.for_project("proj")
            profile = config.resolve_video_workflow_profile(pipeline="minimax_h3", purpose="final")
            assert profile is not None
            self.assertTrue(profile.workflow_path.endswith(".json"))
            self.assertNotEqual("workflows/h3_default.json", profile.workflow_path)
            # other attributes preserved from config default
            self.assertEqual("h3-default", profile.name)
            self.assertEqual(1, profile.stages)

    def test_inactive_import_does_not_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = WorkflowImportStore(projects_root=root)
            store.import_workflow(
                project_id="proj", profile_id="h3-final", pipeline="minimax_h3",
                purpose="final", graph=_GRAPH,
            )
            # still draft — not activated
            config = _make_config(tmp)
            config.import_store = store.for_project("proj")
            profile = config.resolve_video_workflow_profile(pipeline="minimax_h3", purpose="final")
            assert profile is not None
            self.assertEqual("workflows/h3_default.json", profile.workflow_path)

    def test_named_profile_lookup_ignores_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = WorkflowImportStore(projects_root=root)
            store.import_workflow(
                project_id="proj", profile_id="h3-final", pipeline="minimax_h3",
                purpose="final", graph=_GRAPH,
            )
            store.record_validation("proj", "h3-final", valid=True)
            store.record_test_run("proj", "h3-final", TestRunResult(success=True))
            store.activate("proj", "h3-final")

            config = _make_config(tmp)
            config.import_store = store.for_project("proj")
            profile = config.resolve_video_workflow_profile(
                pipeline="minimax_h3", purpose="final", name="h3-default"
            )
            assert profile is not None
            self.assertEqual("workflows/h3_default.json", profile.workflow_path)

    def test_project_scoped_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = WorkflowImportStore(projects_root=root)
            store.import_workflow(
                project_id="proj", profile_id="h3-final", pipeline="minimax_h3",
                purpose="final", graph=_GRAPH,
            )
            store.record_validation("proj", "h3-final", valid=True)
            store.record_test_run("proj", "h3-final", TestRunResult(success=True))
            store.activate("proj", "h3-final")

            view = store.for_project("proj")
            active = view.find_active(pipeline="minimax_h3", purpose="final")
            self.assertIsNotNone(active)
            self.assertTrue(view.is_snapshot_pinned("h3-final") is False)
            path = view.snapshot_path("h3-final")
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
