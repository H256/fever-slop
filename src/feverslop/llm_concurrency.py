"""Process-local LLM concurrency boundary.

This limiter coordinates threads inside one Python process only. It does not
coordinate multiple FeverSlop processes, other clients, or server-side model
slots; those still need server/provider-side limits.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import Condition, Lock
from typing import Any, Iterator


@dataclass(frozen=True, slots=True)
class LLMConcurrencySnapshot:
    max_concurrent_requests: int
    in_flight: int
    max_observed: int


class LLMConcurrencyLimiter:
    def __init__(self, max_concurrent_requests: int = 1) -> None:
        self._max_concurrent_requests = self._validate_limit(max_concurrent_requests)
        self._condition = Condition()
        self._in_flight = 0
        self._max_observed = 0

    @staticmethod
    def _validate_limit(value: int) -> int:
        limit = int(value)
        if limit <= 0:
            raise ValueError("llm.max_concurrent_requests must be > 0")
        return limit

    def configure(self, max_concurrent_requests: int) -> None:
        limit = self._validate_limit(max_concurrent_requests)
        with self._condition:
            self._max_concurrent_requests = limit
            self._condition.notify_all()

    @contextmanager
    def acquire(self) -> Iterator[None]:
        with self._condition:
            while self._in_flight >= self._max_concurrent_requests:
                self._condition.wait()
            self._in_flight += 1
            self._max_observed = max(self._max_observed, self._in_flight)
        try:
            yield
        finally:
            with self._condition:
                self._in_flight -= 1
                self._condition.notify_all()

    def snapshot(self) -> LLMConcurrencySnapshot:
        with self._condition:
            return LLMConcurrencySnapshot(
                max_concurrent_requests=self._max_concurrent_requests,
                in_flight=self._in_flight,
                max_observed=self._max_observed,
            )


_default_limiter_lock = Lock()
_default_limiter = LLMConcurrencyLimiter()
_default_limiter_configured_limit: int | None = None


def get_shared_llm_concurrency_limiter(max_concurrent_requests: int = 1) -> LLMConcurrencyLimiter:
    global _default_limiter_configured_limit
    limit = LLMConcurrencyLimiter._validate_limit(max_concurrent_requests)
    with _default_limiter_lock:
        if _default_limiter_configured_limit is None:
            _default_limiter.configure(limit)
            _default_limiter_configured_limit = limit
        elif _default_limiter_configured_limit != limit:
            raise ValueError(
                "llm.max_concurrent_requests already configured as "
                f"{_default_limiter_configured_limit}; got conflicting value {limit}"
            )
        return _default_limiter


class LimitedDspyLM:
    def __init__(self, lm: Any, limiter: LLMConcurrencyLimiter) -> None:
        self._lm = lm
        self.llm_limiter = limiter

    def __getattr__(self, name: str) -> Any:
        return getattr(self._lm, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with self.llm_limiter.acquire():
            return self._lm(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        with self.llm_limiter.acquire():
            return self._lm.forward(*args, **kwargs)

    def copy(self, **kwargs: Any) -> "LimitedDspyLM":
        copied = self._lm.copy(**kwargs)
        return LimitedDspyLM(copied, self.llm_limiter)


def limit_dspy_lm(lm: Any, limiter: LLMConcurrencyLimiter) -> Any:
    if isinstance(lm, LimitedDspyLM) or not callable(lm):
        return lm
    return LimitedDspyLM(lm, limiter)
