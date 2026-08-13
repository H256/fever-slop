"""Small dependency-free metrics and structured logging helper for API calls."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import logging
from threading import Lock
from time import perf_counter


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
