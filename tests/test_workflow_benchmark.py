from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
from pathlib import Path
import unittest

from feverslop.domain.workflow_benchmark import (
    WorkflowBenchmarkCase,
    WorkflowBenchmarkResult,
)
from feverslop.ports.benchmarking import BenchmarkResultStorePort, MonotonicClockPort


class WorkflowBenchmarkCaseTests(unittest.TestCase):
    def test_creates_immutable_case_with_portable_path(self):
        case = WorkflowBenchmarkCase(
            name=" candidate ",
            prepared_workflow=Path("prepared\\candidate\\workflow.json"),
        )

        self.assertEqual(case.name, "candidate")
        self.assertEqual(case.prepared_workflow.as_posix(), "prepared/candidate/workflow.json")
        with self.assertRaises(FrozenInstanceError):
            case.name = "changed"  # type: ignore[misc]

    def test_rejects_blank_case_name(self):
        with self.assertRaisesRegex(ValueError, "case name"):
            WorkflowBenchmarkCase(name="  ", prepared_workflow=Path("workflow.json"))

    def test_rejects_empty_prepared_workflow_path(self):
        with self.assertRaisesRegex(ValueError, "prepared workflow path"):
            WorkflowBenchmarkCase(name="candidate", prepared_workflow=Path())

    def test_rejects_unsafe_prepared_workflow_paths_portably(self):
        invalid_paths = (
            Path("../workflow.json"),
            Path("/tmp/workflow.json"),
            Path("C:/work/workflow.json"),
            Path("C:\\work\\workflow.json"),
        )

        for prepared_workflow in invalid_paths:
            with self.subTest(prepared_workflow=prepared_workflow):
                with self.assertRaisesRegex(ValueError, "repository-relative"):
                    WorkflowBenchmarkCase(
                        name="candidate",
                        prepared_workflow=prepared_workflow,
                    )


class WorkflowBenchmarkResultTests(unittest.TestCase):
    def setUp(self):
        self.case = WorkflowBenchmarkCase(
            name="candidate",
            prepared_workflow=Path("candidate/workflow.json"),
        )

    def test_rejects_result_with_mismatched_case_name(self):
        with self.assertRaisesRegex(ValueError, "case name"):
            WorkflowBenchmarkResult.create(
                case=self.case,
                reported_case_name="baseline",
                elapsed_seconds=10.0,
                output_path=Path("candidate.mp4"),
            )

    def test_direct_result_construction_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "factory methods"):
            WorkflowBenchmarkResult(
                case_name="candidate",
                prepared_workflow="candidate/workflow.json",
                output_path="candidate.mp4",
                elapsed_seconds=1.0,
                success=True,
            )

    def test_direct_failed_result_cannot_truthiness_coerce_boolean_output_path(self):
        with self.assertRaisesRegex(TypeError, "factory methods"):
            WorkflowBenchmarkResult(
                case_name="candidate",
                prepared_workflow="candidate/workflow.json",
                output_path=False,  # type: ignore[arg-type]
                elapsed_seconds=1.0,
                success=False,
                error="failed",
            )

    def test_successful_result_normalizes_paths_and_values(self):
        result = WorkflowBenchmarkResult.successful(
            self.case,
            Path("output\\candidate.mp4"),
            10,
        )

        self.assertEqual(result.case_name, "candidate")
        self.assertEqual(result.prepared_workflow, "candidate/workflow.json")
        self.assertEqual(result.output_path, "output/candidate.mp4")
        self.assertEqual(result.elapsed_seconds, 10.0)
        self.assertIs(result.success, True)
        self.assertEqual(result.error, "")
        with self.assertRaises(FrozenInstanceError):
            result.success = False  # type: ignore[misc]

    def test_successful_result_requires_positive_finite_elapsed_time(self):
        for elapsed in (0, -1, math.inf, -math.inf, math.nan, True):
            with self.subTest(elapsed=elapsed):
                with self.assertRaisesRegex(ValueError, "elapsed seconds"):
                    WorkflowBenchmarkResult.successful(
                        self.case,
                        Path("candidate.mp4"),
                        elapsed,
                    )

    def test_successful_result_requires_nonempty_output_path(self):
        for output_path in ("", "  ", Path()):
            with self.subTest(output_path=output_path):
                with self.assertRaisesRegex(ValueError, "output path"):
                    WorkflowBenchmarkResult.successful(self.case, output_path, 1.0)

    def test_successful_result_does_not_coerce_boolean_output_path(self):
        with self.assertRaisesRegex(ValueError, "output path"):
            WorkflowBenchmarkResult.successful(self.case, True, 1.0)  # type: ignore[arg-type]

    def test_failed_result_preserves_error_and_accepts_zero_elapsed_time(self):
        error = " render failed: missing model "

        result = WorkflowBenchmarkResult.failed(self.case, 0, error)

        self.assertEqual(result.case_name, "candidate")
        self.assertEqual(result.prepared_workflow, "candidate/workflow.json")
        self.assertEqual(result.output_path, "")
        self.assertEqual(result.elapsed_seconds, 0.0)
        self.assertIs(result.success, False)
        self.assertEqual(result.error, error)

    def test_failed_result_requires_nonnegative_finite_elapsed_time(self):
        for elapsed in (-1, math.inf, -math.inf, math.nan, False):
            with self.subTest(elapsed=elapsed):
                with self.assertRaisesRegex(ValueError, "elapsed seconds"):
                    WorkflowBenchmarkResult.failed(self.case, elapsed, "failed")

    def test_failed_result_requires_nonempty_error(self):
        for error in ("", "  "):
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, "error"):
                    WorkflowBenchmarkResult.failed(self.case, 1.0, error)


class BenchmarkPortTests(unittest.TestCase):
    def test_ports_expose_expected_methods(self):
        self.assertIn("now", MonotonicClockPort.__dict__)
        self.assertIn("write", BenchmarkResultStorePort.__dict__)


if __name__ == "__main__":
    unittest.main()
