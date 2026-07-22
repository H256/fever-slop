from __future__ import annotations

from pathlib import Path
import unittest

from feverslop.application.benchmark_video_workflows import (
    BenchmarkVideoWorkflowsUseCase,
)
from feverslop.domain.workflow_benchmark import WorkflowBenchmarkCase


class FakeClock:
    def __init__(
        self,
        timestamps: list[float],
        events: list[tuple[str, object]] | None = None,
    ) -> None:
        self.timestamps = iter(timestamps)
        self.events = events

    def now(self) -> float:
        timestamp = next(self.timestamps)
        if self.events is not None:
            self.events.append(("clock", timestamp))
        return timestamp


class RecordingResultStore:
    def __init__(self) -> None:
        self.writes = []

    def write(self, results):
        self.writes.append(results)
        return Path("benchmark.json")


class BenchmarkVideoWorkflowsUseCaseTests(unittest.TestCase):
    def test_renders_captures_and_records_successes_in_stable_order(self):
        events: list[tuple[str, object]] = []
        outputs = {
            Path("prepared/baseline.json"): Path("volatile/render.mp4"),
            Path("prepared/candidate.json"): Path("volatile/render.mp4"),
        }

        class Renderer:
            def render(self, prepared_workflow):
                events.append(("render", prepared_workflow))
                return outputs[prepared_workflow]

        class ArtifactStore:
            def capture(self, case_name, rendered_output):
                events.append(("capture", case_name))
                return Path("evidence") / f"{case_name}.mp4"

        cases = (
            WorkflowBenchmarkCase("baseline", Path("prepared/baseline.json")),
            WorkflowBenchmarkCase("candidate", Path("prepared/candidate.json")),
        )
        result_store = RecordingResultStore()
        use_case = BenchmarkVideoWorkflowsUseCase(
            renderer=Renderer(),
            clock=FakeClock([10, 15, 20, 29], events),
            artifact_store=ArtifactStore(),
            result_store=result_store,
        )

        report_path = use_case.execute(cases)

        self.assertEqual(report_path, Path("benchmark.json"))
        self.assertEqual(
            events,
            [
                ("clock", 10),
                ("render", Path("prepared/baseline.json")),
                ("clock", 15),
                ("capture", "baseline"),
                ("clock", 20),
                ("render", Path("prepared/candidate.json")),
                ("clock", 29),
                ("capture", "candidate"),
            ],
        )
        self.assertEqual(len(result_store.writes), 1)
        results = result_store.writes[0]
        self.assertIsInstance(results, tuple)
        self.assertEqual([result.case_name for result in results], ["baseline", "candidate"])
        self.assertEqual([result.elapsed_seconds for result in results], [5.0, 9.0])
        self.assertEqual(
            [result.output_path for result in results],
            ["evidence/baseline.mp4", "evidence/candidate.mp4"],
        )

    def test_failure_is_recorded_and_later_case_still_runs(self):
        captured: list[str] = []

        class Renderer:
            def render(self, prepared_workflow):
                if prepared_workflow.name == "broken.json":
                    raise RuntimeError()
                return Path("volatile/good.mp4")

        class ArtifactStore:
            def capture(self, case_name, rendered_output):
                captured.append(case_name)
                return Path("evidence") / f"{case_name}.mp4"

        cases = (
            WorkflowBenchmarkCase("broken", Path("prepared/broken.json")),
            WorkflowBenchmarkCase("good", Path("prepared/good.json")),
        )
        result_store = RecordingResultStore()
        use_case = BenchmarkVideoWorkflowsUseCase(
            renderer=Renderer(),
            clock=FakeClock([1, 3, 5, 8]),
            artifact_store=ArtifactStore(),
            result_store=result_store,
        )

        use_case.execute(cases)

        self.assertEqual(captured, ["good"])
        failed, successful = result_store.writes[0]
        self.assertFalse(failed.success)
        self.assertEqual(failed.error, "RuntimeError")
        self.assertEqual(failed.elapsed_seconds, 2.0)
        self.assertTrue(successful.success)
        self.assertEqual(successful.elapsed_seconds, 3.0)

    def test_process_control_exceptions_propagate(self):
        for exception_type in (KeyboardInterrupt, SystemExit):
            with self.subTest(exception_type=exception_type):
                class Renderer:
                    def render(self, prepared_workflow):
                        raise exception_type()

                result_store = RecordingResultStore()
                use_case = BenchmarkVideoWorkflowsUseCase(
                    renderer=Renderer(),
                    clock=FakeClock([1]),
                    artifact_store=object(),
                    result_store=result_store,
                )

                with self.assertRaises(exception_type):
                    use_case.execute(
                        (WorkflowBenchmarkCase("case", Path("prepared.json")),)
                    )
                self.assertEqual(result_store.writes, [])


if __name__ == "__main__":
    unittest.main()
