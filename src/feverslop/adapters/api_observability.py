"""Small dependency-free metrics and structured logging helper for API calls."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import logging
from threading import Lock
from time import perf_counter
import time
import re
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


class APIMetrics:
    """Process-local counters suitable for diagnostics and test assertions."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._stats: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0, 0, 0, 0.0])

    def record(self, service: str, operation: str, duration_ms: float, *, success: bool) -> None:
        key = (service, operation)
        with self._lock:
            values = self._stats[key]
            values[0] += 1
            values[1 if success else 2] += 1
            values[3] += duration_ms

    def snapshot(self) -> dict[tuple[str, str], APICallStats]:
        with self._lock:
            return {
                key: APICallStats(int(values[0]), int(values[1]), int(values[2]), values[3])
                for key, values in self._stats.items()
            }


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
) -> None:
    duration_ms = (perf_counter() - started_at) * 1000
    metrics.record(service, operation, duration_ms, success=success)
    if logger is not None:
        logger.info(
            "api_call service=%s operation=%s duration_ms=%.1f success=%s",
            service,
            operation,
            duration_ms,
            str(success).lower(),
        )
