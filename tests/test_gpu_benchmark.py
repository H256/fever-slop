import json
import tempfile
import unittest
from pathlib import Path

from feverslop.tools.gpu_benchmark import (
    BenchmarkRunner,
    FakeBenchmarkBackend,
    SCHEMA,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_SKIPPED,
    expand_matrix,
    write_review_manifest,
)


class _RecordingReporter:
    def __init__(self):
        self.steps = []
        self.messages = []

    def step(self, title):
        self.steps.append(title)

    def message(self, text):
        self.messages.append(text)


class _FailingBackend:
    def render(self, cell):
        raise RuntimeError(f"boom {cell.key}")


class ExpandMatrixTests(unittest.TestCase):
    def test_expands_full_cartesian_product_in_order(self):
        cells = expand_matrix(["a", "b"], [1, 2], [10, 20])
        self.assertEqual(8, len(cells))
        self.assertEqual("a:1:10", cells[0].key)
        self.assertEqual("b:2:20", cells[-1].key)

    def test_requires_nonempty_matrix_dimensions(self):
        with self.assertRaises(ValueError):
            expand_matrix([], [1], [1])
        with self.assertRaises(ValueError):
            expand_matrix(["a"], [], [1])
        with self.assertRaises(ValueError):
            expand_matrix(["a"], [1], [])

    def test_rejects_invalid_profile_scene_seed_types(self):
        with self.assertRaises(ValueError):
            expand_matrix(["  "], [1], [1])
        with self.assertRaises(ValueError):
            expand_matrix(["a"], [0], [1])
        with self.assertRaises(ValueError):
            expand_matrix(["a"], [1], ["x"])


class BenchmarkRunnerTests(unittest.TestCase):
    def _tmp(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)

    def test_renders_every_cell_and_records_timing_vram_inventory(self):
        root = self._tmp()
        backend = FakeBenchmarkBackend(root)
        cells = expand_matrix(["p"], [1], [10, 20])
        reporter = _RecordingReporter()
        runner = BenchmarkRunner(backend, cells, output_dir=root, reporter=reporter)
        manifest = runner.run()

        self.assertEqual(SCHEMA, manifest["schema"])
        self.assertEqual(2, len(manifest["cells"]))
        self.assertEqual(["p"], manifest["matrix"]["profiles"])
        self.assertEqual([1], manifest["matrix"]["scenes"])
        self.assertEqual([10, 20], manifest["matrix"]["seeds"])
        for cell in manifest["cells"]:
            self.assertEqual(STATUS_OK, cell["status"])
            self.assertGreater(cell["duration_seconds"], 0)
            self.assertIsNotNone(cell["peak_vram_mb"])
            self.assertEqual("0" * 64, cell["prepared_workflow_sha256"])
            self.assertTrue(cell["model_inventory"])
        # progress + step boundaries emitted from start
        self.assertTrue(reporter.steps)
        self.assertTrue(any("benchmark cells" in m for m in reporter.messages))

    def test_backend_failure_records_failed_status_and_continues(self):
        root = self._tmp()
        cells = expand_matrix(["p"], [1], [10, 20])
        runner = BenchmarkRunner(_FailingBackend(), cells, output_dir=root)
        manifest = runner.run()
        self.assertEqual(STATUS_FAILED, manifest["cells"][0]["status"])
        self.assertEqual(STATUS_FAILED, manifest["cells"][1]["status"])
        self.assertIn("boom", manifest["cells"][0]["error"])

    def test_resume_skips_completed_cells_and_reuses_recorded_metrics(self):
        root = self._tmp()
        backend = FakeBenchmarkBackend(root)
        cells = expand_matrix(["p"], [1], [10, 20])
        first = BenchmarkRunner(backend, cells, output_dir=root).run()
        self.assertEqual(STATUS_OK, first["cells"][0]["status"])

        calls = []

        class CountingBackend:
            def render(self, cell):
                calls.append(cell.key)
                return backend.render(cell)

        second = BenchmarkRunner(CountingBackend(), cells, output_dir=root).run()
        # all cells already rendered on first pass -> no re-render
        self.assertEqual([], calls)
        self.assertEqual(STATUS_SKIPPED, second["cells"][0]["status"])
        self.assertEqual(STATUS_SKIPPED, second["cells"][1]["status"])
        # skipped cells still carry the recorded metrics
        self.assertEqual(first["cells"][0]["duration_seconds"], second["cells"][0]["duration_seconds"])

    def test_missing_output_file_forces_re_render(self):
        root = self._tmp()
        backend = FakeBenchmarkBackend(root)
        cells = expand_matrix(["p"], [1], [10])
        BenchmarkRunner(backend, cells, output_dir=root).run()
        # delete the output; resume must re-render
        (root / "scene_1_p_seed10.mp4").unlink()
        calls = []

        class CountingBackend:
            def render(self, cell):
                calls.append(cell.key)
                return backend.render(cell)

        second = BenchmarkRunner(CountingBackend(), cells, output_dir=root).run()
        self.assertEqual(["p:1:10"], calls)
        self.assertEqual(STATUS_OK, second["cells"][0]["status"])

    def test_state_file_persists_ok_cells(self):
        root = self._tmp()
        backend = FakeBenchmarkBackend(root)
        cells = expand_matrix(["p"], [1], [10])
        BenchmarkRunner(backend, cells, output_dir=root).run()
        state = json.loads((root / "state.json").read_text())
        self.assertIn("p:1:10", state["cells"])
        self.assertEqual(STATUS_OK, state["cells"]["p:1:10"]["status"])

    def test_write_review_manifest_is_atomic_and_readable(self):
        root = self._tmp()
        manifest = {"schema": SCHEMA, "matrix": {}, "cells": []}
        out = write_review_manifest(manifest, root / "manifest.json")
        self.assertTrue(out.is_file())
        self.assertEqual(manifest, json.loads(out.read_text()))


class CliWiringTests(unittest.TestCase):
    def test_main_renders_matrix_and_writes_manifest(self):
        from feverslop.tools.gpu_benchmark import main

        root = Path(tempfile.mkdtemp())
        self.addCleanup(_rmtree, root)
        rc = main([
            "--profiles", "p",
            "--scenes", "1",
            "--seeds", "10,20",
            "--backend", "fake",
            "--output-dir", str(root),
            "--json",
        ])
        self.assertEqual(0, rc)
        manifest_path = root / "manifest.json"
        self.assertTrue(manifest_path.is_file())
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(2, len(manifest["cells"]))
        self.assertTrue(all(c["status"] == STATUS_OK for c in manifest["cells"]))
        # rendered outputs exist on disk
        for cell in manifest["cells"]:
            self.assertTrue((root / cell["output_path"]).is_file())

    def test_main_requires_all_matrix_dimensions(self):
        from feverslop.tools.gpu_benchmark import main

        with self.assertRaises(SystemExit):
            main(["--profiles", "p", "--output-dir", "x"])


def _rmtree(path: Path) -> None:
    import shutil
    shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
