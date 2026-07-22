from __future__ import annotations

from pathlib import Path

from feverslop.domain.workflow_benchmark import (
    WorkflowBenchmarkCase,
    WorkflowBenchmarkResult,
)
from feverslop.ports.benchmarking import (
    BenchmarkArtifactStorePort,
    BenchmarkResultStorePort,
    MonotonicClockPort,
)
from feverslop.ports.workflow import PreparedWorkflowRendererPort


class BenchmarkVideoWorkflowsUseCase:
    def __init__(
        self,
        *,
        renderer: PreparedWorkflowRendererPort,
        clock: MonotonicClockPort,
        artifact_store: BenchmarkArtifactStorePort,
        result_store: BenchmarkResultStorePort,
    ) -> None:
        self._renderer = renderer
        self._clock = clock
        self._artifact_store = artifact_store
        self._result_store = result_store

    def execute(self, cases: tuple[WorkflowBenchmarkCase, ...]) -> Path:
        results: list[WorkflowBenchmarkResult] = []
        for case in cases:
            started = self._clock.now()
            try:
                rendered_output = self._renderer.render(case.prepared_workflow)
            except Exception as exc:
                ended = self._clock.now()
                error = str(exc)
                if not error.strip():
                    error = type(exc).__name__
                result = WorkflowBenchmarkResult.failed(case, ended - started, error)
            else:
                ended = self._clock.now()
                try:
                    captured_output = self._artifact_store.capture(case.name, rendered_output)
                except Exception as exc:
                    error = str(exc)
                    if not error.strip():
                        error = type(exc).__name__
                    result = WorkflowBenchmarkResult.failed(case, ended - started, error)
                else:
                    result = WorkflowBenchmarkResult.successful(
                        case,
                        captured_output,
                        ended - started,
                    )
            results.append(result)

        return self._result_store.write(tuple(results))
