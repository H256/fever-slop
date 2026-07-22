from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from feverslop.adapters.benchmark_artifacts import LocalBenchmarkArtifactStore


class LocalBenchmarkArtifactStoreTests(unittest.TestCase):
    def test_copies_each_case_to_a_distinct_stable_path(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rendered = root / "volatile.mp4"
            rendered.write_bytes(b"first")
            store = LocalBenchmarkArtifactStore(root / "evidence")

            first = store.capture("baseline", rendered)
            rendered.write_bytes(b"second")
            second = store.capture("candidate", rendered)

            self.assertEqual(first, root / "evidence" / "baseline.mp4")
            self.assertEqual(second, root / "evidence" / "candidate.mp4")
            self.assertEqual(first.read_bytes(), b"first")
            self.assertEqual(second.read_bytes(), b"second")

    def test_rejects_unsafe_case_names_and_does_not_overwrite_evidence(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rendered = root / "render.mp4"
            rendered.write_bytes(b"new")
            store = LocalBenchmarkArtifactStore(root / "evidence")

            for unsafe_name in ("../escape", "nested/case", "nested\\case", ".", ""):
                with self.subTest(unsafe_name=unsafe_name):
                    with self.assertRaises(ValueError):
                        store.capture(unsafe_name, rendered)

            evidence = root / "evidence" / "candidate.mp4"
            evidence.parent.mkdir(parents=True)
            evidence.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                store.capture("candidate", rendered)
            self.assertEqual(evidence.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
