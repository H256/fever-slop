from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from feverslop.domain.workflow_benchmark import WorkflowBenchmarkResult


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JsonBenchmarkResultStore:
    def __init__(
        self,
        output_path: Path,
        *,
        base_path: Path,
        utc_now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._output_path = output_path
        self._base_path = base_path
        self._utc_now = utc_now

    def write(self, results: tuple[WorkflowBenchmarkResult, ...]) -> Path:
        timestamp = self._utc_now().astimezone(timezone.utc)
        payload = {
            "schema": "feverslop.workflow-benchmark/v1",
            "created_at": timestamp.isoformat().replace("+00:00", "Z"),
            "results": [self._serialize_result(result) for result in results],
        }

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self._output_path.parent,
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temporary_path, self._output_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        return self._output_path

    def _serialize_result(self, result: WorkflowBenchmarkResult) -> dict[str, object]:
        return {
            "case_name": result.case_name,
            "prepared_workflow": self._portable_path(result.prepared_workflow),
            "output_path": self._portable_path(result.output_path) if result.output_path else "",
            "elapsed_seconds": result.elapsed_seconds,
            "success": result.success,
            "error": result.error,
        }

    def _portable_path(self, value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            return path.as_posix()
        try:
            return path.resolve().relative_to(self._base_path.resolve()).as_posix()
        except ValueError:
            return path.as_posix()
