from pathlib import Path
from typing import Protocol

from feverslop.domain.workflow_benchmark import WorkflowBenchmarkResult


class MonotonicClockPort(Protocol):
    def now(self) -> float:
        """Return a monotonic timestamp in seconds."""


class BenchmarkResultStorePort(Protocol):
    def write(self, results: tuple[WorkflowBenchmarkResult, ...]) -> Path:
        """Persist one immutable benchmark run."""
