"""Focused tests for P3/P4 render-path wiring: global app_config.json + per-project import.

The config file lives at repo root (base_dir != project dir). The import store
must be re-anchored to the actual project directory so an active import under
``projects/<name>/workflows/`` wins over the config default.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from feverslop.config.app_config import AppConfig
from feverslop.domain.workflow_import import TestRunResult
from feverslop.domain.workflow_import_store import WorkflowImportStore

_GRAPH = {
    "1": {"class_type": "LoadImage", "inputs": {}},
    "2": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {"audio": [3, 0]}},
    "3": {"class_type": "LoadAudio", "inputs": {}},
    "4": {"class_type": "VAEDecode", "inputs": {"samples": [2, 0]}},
}


def _write_config(root: Path) -> Path:
    config_path = root / "app_config.json"
    config_path.write_text(
        json.dumps(
            {
                "execution": {},
                "video_workflow_profiles": [
                    {
                        "name": "h3-default",
                        "pipeline": "minimax_h3",
                        "workflow": "workflows/h3_default.json",
                        "purpose": "final",
                        "stages": 1,
                        "output_scale": 1.0,
                        "supports_per_pass_loras": False,
                        "default": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _activate_import(projects_root: Path, project: str) -> None:
    store = WorkflowImportStore(projects_root=projects_root)
    store.import_workflow(
        project_id=project,
        profile_id="h3-final",
        pipeline="minimax_h3",
        purpose="final",
        graph=_GRAPH,
    )
    store.record_validation(project, "h3-final", valid=True)
    store.record_test_run(project, "h3-final", TestRunResult(success=True))
    store.activate(project, "h3-final")


class RenderPathWiringTests(unittest.TestCase):
    def test_global_config_anchored_to_project_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # app_config.json at repo root; project lives under projects/<name>/
            config_path = _write_config(root)
            project_dir = root / "projects" / "myproj"
            (project_dir / "workflows").mkdir(parents=True)
            _activate_import(root / "projects", "myproj")

            config = AppConfig.load(config_path)
            # base_dir (repo root) has no workflows/ dir -> store not attached yet
            self.assertIsNone(config.import_store)

            # re-anchor to the real project dir (the render-path fix)
            config.attach_import_store(project_dir)

            profile = config.resolve_video_workflow_profile(
                pipeline="minimax_h3", purpose="final"
            )
            assert profile is not None
            # active import wins over the config default
            self.assertNotEqual("workflows/h3_default.json", profile.workflow_path)
            self.assertTrue(profile.workflow_path.endswith(".json"))
            # other attributes preserved from the config default
            self.assertEqual("h3-default", profile.name)
            self.assertEqual(1, profile.stages)

    def test_no_import_keeps_config_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = _write_config(root)
            project_dir = root / "projects" / "empty"
            (project_dir / "workflows").mkdir(parents=True)
            # no import activated

            config = AppConfig.load(config_path)
            config.attach_import_store(project_dir)

            profile = config.resolve_video_workflow_profile(
                pipeline="minimax_h3", purpose="final"
            )
            assert profile is not None
            self.assertEqual("workflows/h3_default.json", profile.workflow_path)

    def test_reanchor_overrides_stale_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = _write_config(root)
            # a stale/wrong project dir with no active import
            stale_dir = root / "projects" / "stale"
            (stale_dir / "workflows").mkdir(parents=True)
            # the real project dir with an active import
            real_dir = root / "projects" / "myproj"
            (real_dir / "workflows").mkdir(parents=True)
            _activate_import(root / "projects", "myproj")

            config = AppConfig.load(config_path)
            config.attach_import_store(stale_dir)  # no import here
            profile = config.resolve_video_workflow_profile(
                pipeline="minimax_h3", purpose="final"
            )
            assert profile is not None
            self.assertEqual("workflows/h3_default.json", profile.workflow_path)

            # re-anchoring must overwrite the stale store (the fix)
            config.attach_import_store(real_dir)
            profile = config.resolve_video_workflow_profile(
                pipeline="minimax_h3", purpose="final"
            )
            assert profile is not None
            self.assertNotEqual("workflows/h3_default.json", profile.workflow_path)
