from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

from feverslop.adapters.json_benchmark_store import JsonBenchmarkResultStore
from feverslop.domain.workflow_benchmark import (
    WorkflowBenchmarkCase,
    WorkflowBenchmarkResult,
)


class JsonBenchmarkResultStoreTests(unittest.TestCase):
    def test_writes_versioned_report_with_utc_timestamp_and_relative_paths(self):
        with TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            report = base / "reports" / "benchmark.json"
            case = WorkflowBenchmarkCase("candidate", base / "prepared" / "workflow.json")
            results = (
                WorkflowBenchmarkResult.successful(
                    case,
                    base / "evidence" / "candidate.mp4",
                    4.5,
                ),
            )
            store = JsonBenchmarkResultStore(
                report,
                base_path=base,
                utc_now=lambda: datetime(2026, 7, 22, 10, 30, tzinfo=timezone.utc),
            )

            written = store.write(results)

            self.assertEqual(written, report)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "feverslop.workflow-benchmark/v1")
            self.assertEqual(payload["created_at"], "2026-07-22T10:30:00Z")
            self.assertEqual(
                payload["results"],
                [
                    {
                        "case_name": "candidate",
                        "prepared_workflow": "prepared/workflow.json",
                        "output_path": "evidence/candidate.mp4",
                        "elapsed_seconds": 4.5,
                        "success": True,
                        "error": "",
                    }
                ],
            )

    def test_preserves_absolute_paths_outside_explicit_base(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            base = root / "base"
            outside = root / "outside"
            result = WorkflowBenchmarkResult.successful(
                WorkflowBenchmarkCase("candidate", outside / "workflow.json"),
                outside / "candidate.mp4",
                1,
            )
            report = base / "benchmark.json"

            JsonBenchmarkResultStore(report, base_path=base).write((result,))

            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["results"][0]["prepared_workflow"],
                (outside / "workflow.json").as_posix(),
            )
            self.assertEqual(
                payload["results"][0]["output_path"],
                (outside / "candidate.mp4").as_posix(),
            )

    def test_refuses_to_overwrite_an_existing_report(self):
        with TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            report = base / "benchmark.json"
            report.write_text("old", encoding="utf-8")
            result = WorkflowBenchmarkResult.failed(
                WorkflowBenchmarkCase("candidate", Path("workflow.json")),
                1,
                "failure",
            )
            store = JsonBenchmarkResultStore(report, base_path=base)

            with self.assertRaises(FileExistsError):
                store.write((result,))

            self.assertEqual(report.read_text(encoding="utf-8"), "old")
            self.assertEqual(list(base.glob("*.tmp")), [])

    def test_create_only_publication_cleans_temporary_file_on_error(self):
        with TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            report = base / "benchmark.json"
            result = WorkflowBenchmarkResult.failed(
                WorkflowBenchmarkCase("candidate", Path("workflow.json")),
                1,
                "failure",
            )
            store = JsonBenchmarkResultStore(report, base_path=base)

            with patch(
                "feverslop.adapters.json_benchmark_store.os.link",
                side_effect=OSError("publish failed"),
            ):
                with self.assertRaisesRegex(OSError, "publish failed"):
                    store.write((result,))

            self.assertFalse(report.exists())
            self.assertEqual(list(base.glob("*.tmp")), [])

    def test_concurrent_writers_publish_exactly_one_report(self):
        with TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            report = base / "benchmark.json"
            result = WorkflowBenchmarkResult.failed(
                WorkflowBenchmarkCase("candidate", Path("workflow.json")),
                1,
                "failure",
            )
            barrier = threading.Barrier(2)
            successes: list[Path] = []
            errors: list[Exception] = []

            def utc_now():
                barrier.wait(timeout=5)
                return datetime(2026, 7, 22, 10, 30, tzinfo=timezone.utc)

            def write_report():
                try:
                    successes.append(
                        JsonBenchmarkResultStore(
                            report, base_path=base, utc_now=utc_now
                        ).write((result,))
                    )
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=write_report) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)

            self.assertEqual(1, len(successes))
            self.assertEqual(1, len(errors))
            self.assertIsInstance(errors[0], FileExistsError)
            self.assertEqual("feverslop.workflow-benchmark/v1", json.loads(report.read_text())["schema"])
            self.assertEqual([], list(base.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
