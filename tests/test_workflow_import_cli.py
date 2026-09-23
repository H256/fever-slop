"""Focused tests for the workflow-import CLI repair subcommand (issue 1317)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rich.console import Console

import main
from feverslop.cli.workflow_import_cli import run_workflow_import_command
from feverslop.domain.workflow_import import TestRunResult
from feverslop.domain.workflow_import_store import WorkflowImportStore

_GOOD_GRAPH = {
    "1": {"class_type": "LoadImage", "inputs": {}},
    "2": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {"audio": [3, 0]}},
    "3": {"class_type": "LoadAudio", "inputs": {}},
    "4": {"class_type": "VAEDecode", "inputs": {"samples": [2, 0]}},
}


class WorkflowImportRepairCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.project_dir = self.root / "proj"
        self.project_dir.mkdir()
        self.store = WorkflowImportStore(projects_root=self.root)
        self.console = Console()

    def _import_and_break(self) -> None:
        self.store.import_workflow(
            project_id="proj", profile_id="h3", pipeline="minimax_h3",
            purpose="final", graph=_GOOD_GRAPH,
        )
        self.store.record_validation("proj", "h3", valid=True)
        self.store.record_test_run(
            "proj", "h3", TestRunResult(success=False, detail="boom")
        )

    def _run_repair(self) -> int:
        args = main.build_arg_parser().parse_args(
            ["workflow-import", "repair",
             "--project-dir", str(self.project_dir), "--profile-id", "h3"]
        )
        return run_workflow_import_command(args, console=self.console)

    def test_parser_exposes_repair(self) -> None:
        args = main.build_arg_parser().parse_args(
            ["workflow-import", "repair",
             "--project-dir", str(self.project_dir), "--profile-id", "h3"]
        )
        self.assertEqual("repair", args.workflow_import_command)

    def test_repair_recovers_broken_to_draft_and_preserves_pins(self) -> None:
        self._import_and_break()
        self.store.pin_snapshot("proj", "h3", "render-1")
        self.assertEqual("broken", self.store.get_profile("proj", "h3").status)
        self.assertEqual(0, self._run_repair())
        self.assertEqual("draft", self.store.get_profile("proj", "h3").status)
        self.assertTrue(self.store.is_snapshot_pinned("proj", "h3"))

    def test_repair_rejects_non_broken_profile(self) -> None:
        self.store.import_workflow(
            project_id="proj", profile_id="h3", pipeline="minimax_h3",
            purpose="final", graph=_GOOD_GRAPH,
        )
        self.assertEqual(1, self._run_repair())
        self.assertEqual("draft", self.store.get_profile("proj", "h3").status)


if __name__ == "__main__":
    unittest.main()
