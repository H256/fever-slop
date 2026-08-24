from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from rich.console import Console

import main
from feverslop.cli.canonical_plan_migration_cli import run_canonical_plan_migration
from feverslop.domain.canonical_render_plan import PromptRole, build_canonical_scene


class CanonicalPlanMigrationCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.project = Path(self.temp_dir.name)
        self.plans = self.project / "output/render/plans"
        self.plans.mkdir(parents=True)
        self.output = io.StringIO()
        self.console = Console(file=self.output, force_terminal=False, color_system=None, width=160)

    def _scene(self) -> dict:
        generated = "generated prompt"
        return {
            "scene": 1,
            "metadata": {"segment_id": "segment-a"},
            "z_image": {"prompt": generated},
            "canonical": build_canonical_scene(
                segment_id="segment-a",
                generated_roles={PromptRole.Z_IMAGE: generated},
            ),
        }

    def _write(self, name: str, value: object) -> None:
        (self.plans / name).write_text(json.dumps(value), encoding="utf-8")

    def _run(self, *extra: str) -> int:
        args = main.build_arg_parser().parse_args(["plan-migrate", str(self.project), *extra])
        return run_canonical_plan_migration(args, console=self.console)

    def test_parser_exposes_dry_run_default_and_explicit_apply(self):
        dry = main.build_arg_parser().parse_args(["plan-migrate", "projects/song"])
        apply = main.build_arg_parser().parse_args(["plan-migrate", "projects/song", "--apply"])

        self.assertEqual("plan-migrate", dry.command)
        self.assertFalse(dry.apply)
        self.assertTrue(apply.apply)

    def test_documented_dry_run_reports_import_without_writing_or_printing_prompt(self):
        scene = self._scene()
        scene["z_image"]["prompt"] = "private human prompt"
        self._write("base.json", [scene])
        before = (self.plans / "base.json").read_bytes()

        exit_code = self._run()

        rendered = self.output.getvalue()
        self.assertEqual(0, exit_code)
        self.assertIn("Dry run", rendered)
        self.assertIn("1 importable", rendered)
        self.assertIn("z_image.prompt", rendered)
        self.assertNotIn("private human prompt", rendered)
        self.assertEqual(before, (self.plans / "base.json").read_bytes())
        self.assertFalse((self.plans / "legacy-migration").exists())

    def test_documented_blocked_conflict_returns_two(self):
        scene = self._scene()
        scene["z_image"]["prompt"] = "base edit"
        self._write("base.json", [scene])
        anchored = json.loads(json.dumps([scene]))
        anchored[0]["z_image"]["prompt"] = "generated prompt"
        references = json.loads(json.dumps([scene]))
        references[0]["z_image"]["prompt"] = "different edit"
        self._write("anchored.json", anchored)
        self._write("references.json", references)

        exit_code = self._run()

        self.assertEqual(2, exit_code)
        self.assertIn("conflicting candidate values", self.output.getvalue())

    def test_apply_writes_override_and_reports_backup(self):
        scene = self._scene()
        scene["z_image"]["prompt"] = "human edit"
        self._write("base.json", [scene])

        exit_code = self._run("--apply")

        self.assertEqual(0, exit_code)
        self.assertIn("Backup:", self.output.getvalue())
        saved = json.loads((self.plans / "base.json").read_text(encoding="utf-8"))
        self.assertEqual(
            "human edit",
            saved[0]["canonical"]["roles"][PromptRole.Z_IMAGE]["override"]["value"],
        )

    def test_missing_or_corrupt_base_returns_one(self):
        self.assertEqual(1, self._run())
        self.assertIn("Canonical base plan does not exist", self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
