import argparse
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from feverslop.adapters.global_library import GlobalLibraryAdapter
from feverslop.domain.global_library import AssetKind, AssetLook, GlobalAsset
from feverslop.tools.global_library_cli import build_arg_parser, main


class GlobalLibraryCliTests(unittest.TestCase):
    def test_create_list_and_show_json(self):
        with tempfile.TemporaryDirectory() as temp:
            args = ["--library-root", temp, "create", "--kind", "prop", "--id", "lamp", "--name", "Lamp"]
            self.assertEqual(0, main(args))
            human_output = StringIO()
            with redirect_stdout(human_output):
                self.assertEqual(0, main(["--library-root", temp, "show", "--kind", "prop", "--id", "lamp"]))
            self.assertEqual("prop/lamp: Lamp\n", human_output.getvalue())
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, main(["--library-root", temp, "list", "--json"]))
            self.assertEqual("lamp", json.loads(output.getvalue())[0]["id"])
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, main(["--library-root", temp, "show", "--kind", "prop", "--id", "lamp", "--json"]))
            self.assertEqual("lamp", json.loads(output.getvalue())["id"])

    def test_invalid_input_returns_nonzero_and_actionable_error(self):
        with tempfile.TemporaryDirectory() as temp:
            error = StringIO()
            with redirect_stderr(error):
                code = main(["--library-root", temp, "show", "--kind", "prop", "--id", "missing"])
            self.assertNotEqual(0, code)
            self.assertIn("create or import", error.getvalue())

    def test_generate_dry_run_accepts_json_input_and_explicit_workflow(self):
        with tempfile.TemporaryDirectory() as temp:
            input_path = Path(temp) / "idea.json"
            input_path.write_text(json.dumps({
                "kind": "character", "asset_id": "ava", "name": "Ava", "visual_concept": "silver bob",
            }), encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, main([
                    "--library-root", temp, "generate", "--input", str(input_path),
                    "--workflow", "character-sheet-v1", "--dry-run",
                ]))
            payload = json.loads(output.getvalue())
            self.assertEqual("ava", payload["asset_id"])
            self.assertEqual("character-sheet-v1", payload["workflow_profile"])

    def test_prune_subcommand_removed(self):
        parser = build_arg_parser()
        subparsers = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
        self.assertNotIn("prune", subparsers.choices)
        with self.assertRaises(SystemExit) as ctx:
            build_arg_parser().parse_args(["prune"])
        self.assertEqual(2, ctx.exception.code)

    def test_refresh_rejects_shallow_snapshot_layout(self):
        temp, library_root, snapshot = self._snapshot_fixture()
        snap = temp / "proj2" / "snap"
        if len(snap.resolve().parents) >= 5:
            self.skipTest("temp directory is too deep to exercise the shallow-snapshot guard")
        snap.mkdir(parents=True)
        (snap / "manifest.json").write_text((snapshot / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8")
        error = StringIO()
        with redirect_stdout(StringIO()), redirect_stderr(error):
            code = main(["--library-root", str(library_root), "refresh", "--snapshot", str(snap)])
        self.assertEqual(2, code)
        self.assertIn("refusing to materialize", error.getvalue())
        self.assertTrue((snap / "manifest.json").is_file())
        self.assertTrue((snapshot / "manifest.json").is_file())
        self.assertTrue((snapshot / "hero.png").is_file())

    def test_refresh_canonical_snapshot_is_a_noop_move(self):
        temp, library_root, snapshot = self._snapshot_fixture()
        manifest_before = (snapshot / "manifest.json").read_text(encoding="utf-8")
        media_before = (snapshot / "hero.png").read_bytes()
        output = StringIO()
        with redirect_stdout(output), redirect_stderr(StringIO()):
            code = main(["--library-root", str(library_root), "refresh", "--snapshot", str(snapshot)])
        self.assertEqual(0, code)
        self.assertIn("refreshed", output.getvalue())
        self.assertEqual(manifest_before, (snapshot / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(media_before, (snapshot / "hero.png").read_bytes())

    def test_create_look_verifies_media_existence(self):
        temp, library_root, snapshot = self._snapshot_fixture()
        command = [
            "--library-root", str(library_root), "create-look",
            "--kind", "character", "--id", "ava",
            "--look-id", "costume", "--name", "Costume",
            "--hero-image", "looks/costume/hero.png",
        ]
        error = StringIO()
        with redirect_stdout(StringIO()), redirect_stderr(error):
            code = main(command)
        self.assertEqual(2, code)
        message = error.getvalue()
        self.assertIn("hero_image", message)
        self.assertIn("looks/costume/hero.png", message)
        media = library_root / "character" / "ava" / "looks" / "costume" / "hero.png"
        media.parent.mkdir(parents=True)
        media.write_bytes(b"costume-hero")
        output = StringIO()
        with redirect_stdout(output), redirect_stderr(StringIO()):
            code = main(command)
        self.assertEqual(0, code)
        self.assertIn("created look character/ava/costume", output.getvalue())

    def test_validate_reports_dangling_media(self):
        temp, library_root, snapshot = self._snapshot_fixture()
        output = StringIO()
        with redirect_stdout(output), redirect_stderr(StringIO()):
            code = main(["--library-root", str(library_root), "validate", "--kind", "character"])
        self.assertEqual(0, code)
        self.assertIn("validated 1 asset manifest(s)", output.getvalue())
        (library_root / "character" / "ava" / "looks" / "base" / "hero.png").unlink()
        error = StringIO()
        with redirect_stdout(StringIO()), redirect_stderr(error):
            code = main(["--library-root", str(library_root), "validate", "--kind", "character"])
        self.assertEqual(2, code)
        message = error.getvalue()
        self.assertIn("hero_image", message)
        self.assertIn("looks/base/hero.png", message)

    def _snapshot_fixture(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        temp = Path(temp_dir.name)
        library_root = temp / "library"
        adapter = GlobalLibraryAdapter(library_root)
        media_rel = "looks/base/hero.png"
        asset = GlobalAsset(
            "ava", AssetKind.CHARACTER, "Ava",
            looks=(AssetLook("base", "Base", hero_image=media_rel),),
        )
        adapter.create(asset)
        media_path = library_root / "character" / "ava" / media_rel
        media_path.parent.mkdir(parents=True)
        media_path.write_bytes(b"base-hero")
        snapshot = adapter.materialize("character", "ava", "base", temp / "proj")
        return temp, library_root, snapshot


if __name__ == "__main__":
    unittest.main()
