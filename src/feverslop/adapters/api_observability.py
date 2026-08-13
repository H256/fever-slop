"""Small dependency-free metrics and structured logging helper for API calls."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import logging
from threading import Lock
from time import perf_counter
import re


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


@dataclass(frozen=True)
class APICallStats:
    calls: int
    successes: int
    failures: int
    total_duration_ms: float
    usage_units: float
    estimated_cost: float


class APIMetrics:
    """Process-local counters suitable for diagnostics and test assertions."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._stats: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0, 0, 0, 0.0, 0.0, 0.0])

    def record(self, service: str, operation: str, duration_ms: float, *, success: bool, usage_units: float = 0.0, estimated_cost: float = 0.0) -> None:
        key = (service, operation)
        with self._lock:
            values = self._stats[key]
            values[0] += 1
            values[1 if success else 2] += 1
            values[3] += duration_ms
            values[4] += float(usage_units)
            values[5] += float(estimated_cost)

    def snapshot(self) -> dict[tuple[str, str], APICallStats]:
        with self._lock:
            return {
                key: APICallStats(int(values[0]), int(values[1]), int(values[2]), values[3], values[4], values[5])
                for key, values in self._stats.items()
            }


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
) -> None:
    duration_ms = (perf_counter() - started_at) * 1000
    metrics.record(service, operation, duration_ms, success=success, usage_units=usage_units, estimated_cost=estimated_cost)
    if logger is not None:
        logger.info(
            "api_call service=%s operation=%s duration_ms=%.1f success=%s",
            service,
            operation,
            duration_ms,
            str(success).lower(),
        )
