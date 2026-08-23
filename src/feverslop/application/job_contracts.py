"""Transport-neutral contracts for application-owned job orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from feverslop.domain.visual_consistency import PreflightMode


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class JobSubmission:
    project_id: str
    action: str
    project_type: str = "standard_music_video"
    pipeline_mode: str | None = None
    reject_if_project_active: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JobRequest:
    """Application request shared by headless job services."""

    action: str
    scenes: list[int] | None = None
    pipeline_mode: str | None = None
    thumbnails: list[dict[str, Any]] | None = None
    reference_kind: str | None = None
    reference_id: str | None = None
    raw_clip: str | None = None
    output_clip: str | None = None
    raw_in_seconds: float | None = None
    raw_out_seconds: float | None = None
    exact: bool = False
    plan: str | None = None
    visual_consistency_mode: str | None = None
    workflow_profile: str | None = None
    preflight_mode: PreflightMode = PreflightMode.WARN

    def __post_init__(self) -> None:
        object.__setattr__(self, "preflight_mode", PreflightMode.parse(self.preflight_mode))


@dataclass(frozen=True)
class JobLogEvent:
    job_id: str
    message: str
    sequence: int = 0
    level: str = "info"


@dataclass(frozen=True)
class JobSnapshot:
    job_id: str
    project_id: str
    action: str
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0
    overall_progress: int = 0
    current_step: str | None = None
    error: str | None = None
    result: Any = None
    logs: tuple[JobLogEvent, ...] = ()


class LogSink(Protocol):
    def __call__(self, message: str) -> None:
        ...


JobHandler = Callable[[LogSink], Any]


@runtime_checkable
class JobRuntime(Protocol):
    def submit(self, submission: JobSubmission, handler: JobHandler) -> str:
        ...

    def get(self, job_id: str) -> JobSnapshot:
        ...

    def list(self, project_id: str | None = None) -> tuple[JobSnapshot, ...]:
        ...

    def cancel(self, job_id: str) -> bool:
        ...

    def shutdown(self, *, wait: bool = True, cancel_pending: bool = False) -> None:
        ...
