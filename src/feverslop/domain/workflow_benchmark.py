from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from pathlib import Path, PurePosixPath, PureWindowsPath


def _portable_path(value: str | Path, *, label: str, repository_relative: bool) -> str:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{label} must be a path")

    raw = str(value).strip()
    if not raw or raw == ".":
        raise ValueError(f"{label} is required")

    windows_path = PureWindowsPath(raw)
    posix_path = PurePosixPath(raw.replace("\\", "/"))
    if repository_relative and (
        windows_path.drive
        or windows_path.root
        or posix_path.is_absolute()
        or ".." in posix_path.parts
    ):
        raise ValueError(f"{label} must be repository-relative")
    return posix_path.as_posix()


def _elapsed_seconds(value: float, *, successful: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("elapsed seconds must be a finite number")

    elapsed = float(value)
    if not isfinite(elapsed) or elapsed < 0 or (successful and elapsed == 0):
        qualifier = "positive" if successful else "nonnegative"
        raise ValueError(f"elapsed seconds must be finite and {qualifier}")
    return elapsed


@dataclass(frozen=True)
class WorkflowBenchmarkCase:
    name: str
    prepared_workflow: Path

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("benchmark case name is required")
        if not isinstance(self.prepared_workflow, Path):
            raise ValueError("prepared workflow path must be a Path")

        normalized_path = _portable_path(
            self.prepared_workflow,
            label="prepared workflow path",
            repository_relative=True,
        )
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "prepared_workflow", Path(normalized_path))


@dataclass(frozen=True)
class WorkflowBenchmarkResult:
    case_name: str
    prepared_workflow: str
    output_path: str
    elapsed_seconds: float
    success: bool
    error: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.case_name, str) or not self.case_name.strip():
            raise ValueError("benchmark case name is required")
        if type(self.success) is not bool:
            raise ValueError("benchmark success must be a boolean")
        if not isinstance(self.error, str):
            raise ValueError("benchmark error must be a string")

        prepared_workflow = _portable_path(
            self.prepared_workflow,
            label="prepared workflow path",
            repository_relative=True,
        )
        elapsed = _elapsed_seconds(self.elapsed_seconds, successful=self.success)

        if self.success:
            output_path = _portable_path(
                self.output_path,
                label="benchmark output path",
                repository_relative=False,
            )
            if self.error:
                raise ValueError("successful benchmark result cannot contain an error")
        else:
            if self.output_path:
                raise ValueError("failed benchmark result cannot contain an output path")
            if not self.error.strip():
                raise ValueError("failed benchmark result requires an error")
            output_path = ""

        object.__setattr__(self, "case_name", self.case_name.strip())
        object.__setattr__(self, "prepared_workflow", prepared_workflow)
        object.__setattr__(self, "output_path", output_path)
        object.__setattr__(self, "elapsed_seconds", elapsed)

    @classmethod
    def create(
        cls,
        *,
        case: WorkflowBenchmarkCase,
        reported_case_name: str,
        elapsed_seconds: float,
        output_path: str | Path,
    ) -> WorkflowBenchmarkResult:
        if not isinstance(reported_case_name, str) or reported_case_name.strip() != case.name:
            raise ValueError("reported case name does not match benchmark case name")
        return cls.successful(case, output_path, elapsed_seconds)

    @classmethod
    def successful(
        cls,
        case: WorkflowBenchmarkCase,
        output_path: str | Path,
        elapsed_seconds: float,
    ) -> WorkflowBenchmarkResult:
        normalized_output_path = _portable_path(
            output_path,
            label="benchmark output path",
            repository_relative=False,
        )
        return cls(
            case_name=case.name,
            prepared_workflow=case.prepared_workflow.as_posix(),
            output_path=normalized_output_path,
            elapsed_seconds=elapsed_seconds,
            success=True,
        )

    @classmethod
    def failed(
        cls,
        case: WorkflowBenchmarkCase,
        elapsed_seconds: float,
        error: str,
    ) -> WorkflowBenchmarkResult:
        return cls(
            case_name=case.name,
            prepared_workflow=case.prepared_workflow.as_posix(),
            output_path="",
            elapsed_seconds=elapsed_seconds,
            success=False,
            error=error,
        )
