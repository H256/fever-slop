from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from feverslop.tools.benchmark_fixture import validate_benchmark_project


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkFixtureTests(unittest.TestCase):
    def test_tracked_example_project_has_portable_verified_manifest(self):
        manifest = validate_benchmark_project(ROOT / "example-project")
        self.assertEqual("3.12", manifest["toolchain"]["python"])
        self.assertTrue(manifest["outputs"]["gitignored"])

    def test_input_hash_mismatch_is_rejected(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "input").mkdir()
            (root / "input/song.mp3").write_bytes(b"fixture")
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "benchmark.json").write_text(json.dumps({
                "schema": "feverslop.benchmark-project/v1",
                "project": {
                    "config": "config.json",
                    "input_audio": "input/song.mp3",
                    "input_audio_sha256": "0" * 64,
                },
                "toolchain": {"python": "3.12"},
                "expected_planning_artifacts": [],
                "outputs": {"root": "output/", "gitignored": True},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_benchmark_project(root)


if __name__ == "__main__":
    unittest.main()
