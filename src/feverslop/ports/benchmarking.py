from pathlib import Path
from typing import Protocol

from feverslop.domain.workflow_benchmark import WorkflowBenchmarkResult


class MonotonicClockPort(Protocol):
    def now(self) -> float:
        """Return a monotonic timestamp in seconds."""


class BenchmarkArtifactStorePort(Protocol):
    def capture(self, case_name: str, rendered_output: Path) -> Path:
        """Preserve one rendered output under its distinct benchmark case identity."""


class BenchmarkResultStorePort(Protocol):
    def write(self, results: tuple[WorkflowBenchmarkResult, ...]) -> Path:
        """Persist one immutable benchmark run."""
