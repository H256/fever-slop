"""Small dependency-free metrics and structured logging helper for API calls."""

from __future__ import annotations

import json
import logging
import math
import re
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from typing import Any

_SENSITIVE_URL_PART = re.compile(
    r"(?P<key>(?:api[_-]?key|access[_-]?token|auth(?:orization)?|bearer|password|secret|token))"
    r"(?P<sep>\s*[:=]\s*|\s+)(?P<value>[^\s&;,]+)",
    re.IGNORECASE,
)
_SENSITIVE_QUERY_PART = re.compile(
    r"(?P<key>(?:api[_-]?key|access[_-]?token|auth(?:orization)?|password|secret|token))"
    r"=(?P<value>[^&\s]+)",
    re.IGNORECASE,
)
_BEARER_TOKEN = re.compile(r"\bBearer\s+[^\s&;,]+", re.IGNORECASE)


def redact_secrets(value: object) -> str:
    """Redact credential-like values before they enter logs or exceptions."""
    text = str(value)
    text = _BEARER_TOKEN.sub("Bearer [REDACTED]", text)
    text = _SENSITIVE_QUERY_PART.sub(lambda match: f"{match.group('key')}=[REDACTED]", text)
    return _SENSITIVE_URL_PART.sub(lambda match: f"{match.group('key')}{match.group('sep')}[REDACTED]", text)


def require_json_object(payload: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{context} response must be a JSON object")
    return payload


@dataclass(frozen=True)
class APICallStats:
    calls: int
    successes: int
    failures: int
    total_duration_ms: float
    usage_units: float
    estimated_cost: float
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    retry_attempts: int = 0
    p50_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    p99_duration_ms: float = 0.0
    correlation_ids: tuple[str, ...] = ()
    job_ids: tuple[str, ...] = ()
    project_ids: tuple[str, ...] = ()
    scene_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class APIObservabilityContext:
    correlation_id: str
    job_id: str | None = None
    project_id: str | None = None
    scene_id: str | None = None


_observability_context: ContextVar[APIObservabilityContext | None] = ContextVar(
    "feverslop_api_observability_context", default=None,
)


@contextmanager
def api_observability_context(
    *,
    correlation_id: str | None = None,
    job_id: str | None = None,
    project_id: str | None = None,
    scene_id: str | None = None,
):
    context = APIObservabilityContext(
        correlation_id=correlation_id or uuid.uuid4().hex,
        job_id=job_id,
        project_id=project_id,
        scene_id=scene_id,
    )
    token = _observability_context.set(context)
    try:
        yield context
    finally:
        _observability_context.reset(token)


@dataclass(frozen=True)
class _APICallSample:
    timestamp: float
    duration_ms: float
    success: bool
    usage_units: float
    estimated_cost: float
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    retry_attempts: int
    correlation_id: str
    job_id: str | None
    project_id: str | None
    scene_id: str | None


class APIMetrics:
    """Process-local counters suitable for diagnostics and test assertions."""

    def __init__(self, *, max_samples_per_operation: int = 10_000) -> None:
        if max_samples_per_operation < 1:
            raise ValueError("max_samples_per_operation must be at least 1")
        self._lock = Lock()
        self._max_samples_per_operation = int(max_samples_per_operation)
        self._totals: dict[tuple[str, str], list[float]] = defaultdict(
            lambda: [0, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0, 0],
        )
        self._samples: dict[tuple[str, str], list[_APICallSample]] = defaultdict(list)

    def record(
        self,
        service: str,
        operation: str,
        duration_ms: float,
        *,
        success: bool,
        usage_units: float = 0.0,
        estimated_cost: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        reasoning_tokens: int = 0,
        retry_attempts: int = 0,
        correlation_id: str | None = None,
        job_id: str | None = None,
        project_id: str | None = None,
        scene_id: str | None = None,
        timestamp: float | None = None,
    ) -> None:
        key = (service, operation)
        context = _observability_context.get()
        sample = _APICallSample(
            timestamp=time.time() if timestamp is None else float(timestamp),
            duration_ms=float(duration_ms),
            success=bool(success),
            usage_units=float(usage_units),
            estimated_cost=float(estimated_cost),
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            reasoning_tokens=int(reasoning_tokens),
            retry_attempts=int(retry_attempts),
            correlation_id=(correlation_id or (context.correlation_id if context else None) or uuid.uuid4().hex),
            job_id=job_id if job_id is not None else (context.job_id if context else None),
            project_id=project_id if project_id is not None else (context.project_id if context else None),
            scene_id=scene_id if scene_id is not None else (context.scene_id if context else None),
        )
        with self._lock:
            totals = self._totals[key]
            totals[0] += 1
            totals[1 if sample.success else 2] += 1
            totals[3] += sample.duration_ms
            totals[4] += sample.usage_units
            totals[5] += sample.estimated_cost
            totals[6] += sample.prompt_tokens
            totals[7] += sample.completion_tokens
            totals[8] += sample.reasoning_tokens
            totals[9] += sample.retry_attempts
            samples = self._samples[key]
            samples.append(sample)
            if len(samples) > self._max_samples_per_operation:
                del samples[: len(samples) - self._max_samples_per_operation]

    def snapshot(self, *, window_seconds: float | None = None, now: float | None = None) -> dict[tuple[str, str], APICallStats]:
        if window_seconds is not None and window_seconds < 0:
            raise ValueError("window_seconds must be non-negative")
        current_time = time.time() if now is None else float(now)
        with self._lock:
            snapshots = {}
            for key, samples in self._samples.items():
                if window_seconds is None:
                    selected = list(samples)
                else:
                    cutoff = current_time - window_seconds
                    selected = [sample for sample in samples if sample.timestamp >= cutoff]
                if selected:
                    snapshots[key] = _summarize_samples(
                        selected,
                        totals=None if window_seconds is not None else self._totals[key],
                    )
            return snapshots

    def export_snapshot(self, *, window_seconds: float | None = None, now: float | None = None) -> dict:
        """Return a stable, JSON-serializable observability snapshot."""
        rows = []
        for (service, operation), stats in sorted(self.snapshot(window_seconds=window_seconds, now=now).items()):
            rows.append({
                "service": service,
                "operation": operation,
                "calls": stats.calls,
                "successes": stats.successes,
                "failures": stats.failures,
                "total_duration_ms": stats.total_duration_ms,
                "usage_units": stats.usage_units,
                "estimated_cost": stats.estimated_cost,
                "prompt_tokens": stats.prompt_tokens,
                "completion_tokens": stats.completion_tokens,
                "reasoning_tokens": stats.reasoning_tokens,
                "retry_attempts": stats.retry_attempts,
                "p50_duration_ms": stats.p50_duration_ms,
                "p95_duration_ms": stats.p95_duration_ms,
                "p99_duration_ms": stats.p99_duration_ms,
                "correlation_id": stats.correlation_ids[-1] if stats.correlation_ids else "",
                "correlation_ids": list(stats.correlation_ids),
                "job_ids": list(stats.job_ids),
                "project_ids": list(stats.project_ids),
                "scene_ids": list(stats.scene_ids),
                "job_id": stats.job_ids[-1] if stats.job_ids else None,
                "project_id": stats.project_ids[-1] if stats.project_ids else None,
                "scene_id": stats.scene_ids[-1] if stats.scene_ids else None,
            })
        return {"version": 1, "entries": rows}

    def export_json(self, *, window_seconds: float | None = None, now: float | None = None) -> str:
        return json.dumps(
            self.export_snapshot(window_seconds=window_seconds, now=now),
            ensure_ascii=False,
            sort_keys=True,
        )


class RequestRateLimiter:
    """Thread-safe minimum-interval limiter; disabled when interval is zero."""

    def __init__(self, min_interval_seconds: float = 0.0):
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must be non-negative")
        self.min_interval_seconds = float(min_interval_seconds)
        self._lock = Lock()
        self._last_request = 0.0

    def wait(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delay = self.min_interval_seconds - (now - self._last_request)
            if delay > 0:
                time.sleep(delay)
                now = time.monotonic()
            self._last_request = now


default_api_metrics = APIMetrics()


def record_api_call(
    metrics: APIMetrics,
    logger: logging.Logger | None,
    service: str,
    operation: str,
    started_at: float,
    *,
    success: bool,
    usage_units: float = 0.0,
    estimated_cost: float = 0.0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    reasoning_tokens: int = 0,
    retry_attempts: int = 0,
    correlation_id: str | None = None,
) -> None:
    duration_ms = (perf_counter() - started_at) * 1000
    context = _observability_context.get()
    resolved_correlation_id = correlation_id or (context.correlation_id if context else None) or uuid.uuid4().hex
    metrics.record(
        service,
        operation,
        duration_ms,
        success=success,
        usage_units=usage_units,
        estimated_cost=estimated_cost,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        retry_attempts=retry_attempts,
        correlation_id=resolved_correlation_id,
    )
    if logger is not None:
        logger.info(
            "api_call service=%s operation=%s duration_ms=%.1f success=%s correlation_id=%s retry_attempts=%d",
            service,
            operation,
            duration_ms,
            str(success).lower(),
            resolved_correlation_id or "",
            retry_attempts,
        )


def _summarize_samples(samples: list[_APICallSample], totals: list[float] | None = None) -> APICallStats:
    durations = sorted(sample.duration_ms for sample in samples)
    values = totals or [
        len(samples),
        sum(sample.success for sample in samples),
        sum(not sample.success for sample in samples),
        sum(sample.duration_ms for sample in samples),
        sum(sample.usage_units for sample in samples),
        sum(sample.estimated_cost for sample in samples),
        sum(sample.prompt_tokens for sample in samples),
        sum(sample.completion_tokens for sample in samples),
        sum(sample.reasoning_tokens for sample in samples),
        sum(sample.retry_attempts for sample in samples),
    ]
    return APICallStats(
        calls=int(values[0]),
        successes=int(values[1]),
        failures=int(values[2]),
        total_duration_ms=values[3],
        usage_units=values[4],
        estimated_cost=values[5],
        prompt_tokens=int(values[6]),
        completion_tokens=int(values[7]),
        reasoning_tokens=int(values[8]),
        retry_attempts=int(values[9]),
        p50_duration_ms=_percentile(durations, 0.50),
        p95_duration_ms=_percentile(durations, 0.95),
        p99_duration_ms=_percentile(durations, 0.99),
        correlation_ids=tuple(dict.fromkeys(sample.correlation_id for sample in samples)),
        job_ids=tuple(dict.fromkeys(sample.job_id for sample in samples if sample.job_id)),
        project_ids=tuple(dict.fromkeys(sample.project_id for sample in samples if sample.project_id)),
        scene_ids=tuple(dict.fromkeys(sample.scene_id for sample in samples if sample.scene_id)),
    )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, math.ceil(percentile * len(values)) - 1))
    return float(values[index])
