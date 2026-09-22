"""Tests for JsonArtifactStore atomic write semantics (issue 1320)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import feverslop.utils.io as io_mod
from feverslop.adapters.local_artifacts import JsonArtifactStore


class JsonArtifactStoreAtomicWriteTests(unittest.TestCase):
    def test_write_render_plan_is_atomic_and_roundtrips(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "render_plan.json"
            store = JsonArtifactStore()
            store.write_render_plan(path, [{"scene_number": 1, "seed": 42}])
            self.assertTrue(path.is_file())
            self.assertEqual([], list(Path(d).glob("*.tmp")))
            self.assertEqual(
                [{"scene_number": 1, "seed": 42}], store.read_render_plan(path)
            )

    def test_write_render_plan_leaves_no_truncated_file_on_midwrite_failure(self):
        """A failure during the write must not truncate the existing plan."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "render_plan.json"
            store = JsonArtifactStore()
            store.write_render_plan(path, [{"scene_number": 1, "seed": 1}])
            original = path.read_text()

            def failing_write(p, payload, **kw):
                # Simulate a crash after opening the temp file but before
                # the atomic replace: the target file must stay untouched.
                p = Path(p)
                p.parent.mkdir(parents=True, exist_ok=True)
                raise OSError("crash mid-write")

            with patch.object(io_mod, "_write_atomic", side_effect=failing_write):
                with self.assertRaises(OSError):
                    store.write_render_plan(path, [{"scene_number": 1, "seed": 999}])
            self.assertEqual(original, path.read_text())

    def test_write_text_is_atomic_and_no_tmp_left(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "notes.txt"
            store = JsonArtifactStore()
            store.write_text(path, "hello")
            self.assertTrue(path.is_file())
            self.assertEqual([], list(Path(d).glob("*.tmp")))
            self.assertEqual("hello", store.read_text(path))

    def test_write_json_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a" / "b" / "data.json"
            store = JsonArtifactStore()
            store.write_json(path, {"ok": True})
            self.assertTrue(path.is_file())
            self.assertEqual({"ok": True}, json.loads(path.read_text()))


if __name__ == "__main__":
    unittest.main()
