"""Shared test doubles for LLM ports.

Replaces duplicated FakeLLM/FakeVisionLLM definitions scattered across
individual test files. Import from here instead of defining inline.

    from tests.fakellm import FakeLLM, FakeVisionLLM, FailingLLM, FailingVisionLLM,\
                             FailingVisionAndTextLLM, VisionOnlyLLM
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CallRecord:
    """Structured record of a single LLM call."""

    system_prompt: str
    prompt: str | None = None
    image_paths: list[Path] | None = None
    timeout: float | None = None


class FakeLLM:
    """Shared test double for the text LLM port.

    Supports three mutually exclusive response modes:

      - ``response: str``      -- single canned response for every call
      - ``responses: list[str]`` -- sequential responses (IndexError on exhaustion)
      - ``side_effect: Callable`` -- per-call customization

    All calls are recorded in ``self.calls`` as :class:`CallRecord` objects.
    """

    def __init__(
        self,
        response: str | None = None,
        responses: list[str] | None = None,
        *,
        side_effect: Callable[..., Any] | None = None,
    ):
        self.calls: list[CallRecord] = []

        if side_effect is not None:
            self._mode: str = "side_effect"
            self._side_effect: Callable[..., Any] = side_effect
        elif isinstance(response, str):
            self._mode = "single"
            self._canned: str = response
        elif responses is not None:
            self._mode = "sequence"
            self._canned_seq: list[str] = list(responses)
        else:
            self._mode = "empty"
            self._canned = ""

    def complete_prompt(
        self,
        system_prompt: str,
        prompt: str,
        timeout: float | None = None,
    ) -> str:
        rec = CallRecord(system_prompt=system_prompt, prompt=prompt, timeout=timeout)
        self.calls.append(rec)
        return self._next_response(system_prompt, prompt)

    def _next_response(self, system_prompt: str, prompt: str) -> str:
        if self._mode == "single":
            return self._canned
        if self._mode == "sequence":
            idx = len(self.calls) - 1
            if idx < 0 or idx >= len(self._canned_seq):
                raise IndexError(
                    f"FakeLLM response sequence exhausted "
                    f"(requested call {idx + 1}, only {len(self._canned_seq)} provided)",
                )
            return self._canned_seq[idx]
        if self._mode == "side_effect" and self._side_effect is not None:
            return self._side_effect(system_prompt, prompt)
        return self._canned


class FakeVisionLLM(FakeLLM):
    """FakeLLM that also implements the vision LLM port.

    Shares the same response mechanism as ``FakeLLM`` for both text and
    vision calls. Use ``FailingVisionLLM`` if you need different behavior
    per port.
    """

    def complete_prompt_with_images(
        self,
        system_prompt: str,
        prompt: str,
        image_paths: list[Path],
        timeout: float | None = None,
    ) -> str:
        rec = CallRecord(
            system_prompt=system_prompt,
            prompt=prompt,
            image_paths=image_paths,
            timeout=timeout,
        )
        self.calls.append(rec)
        return self._next_response(system_prompt, prompt)


class FailingLLM(FakeLLM):
    """FakeLLM that raises on every text completion."""

    def __init__(self, error: Exception = RuntimeError("LLM failed")):
        super().__init__()
        self._error = error

    def complete_prompt(
        self,
        system_prompt: str,
        prompt: str,
        timeout: float | None = None,
    ) -> str:
        self.calls.append(CallRecord(system_prompt=system_prompt, prompt=prompt, timeout=timeout))
        raise self._error


class FailingVisionLLM(FakeLLM):
    """FakeLLM that raises on vision calls but works for text."""

    def __init__(self, response: str = ""):
        super().__init__(response)

    def complete_prompt_with_images(
        self,
        system_prompt: str,
        prompt: str,
        image_paths: list[Path],
        timeout: float | None = None,
    ) -> str:
        self.calls.append(
            CallRecord(
                system_prompt=system_prompt,
                prompt=prompt,
                image_paths=image_paths,
                timeout=timeout,
            ),
        )
        raise RuntimeError("vision failed")


class FailingVisionAndTextLLM(FailingVisionLLM):
    """Raises RuntimeError on both text and vision calls."""

    def complete_prompt(
        self,
        system_prompt: str,
        prompt: str,
        timeout: float | None = None,
    ) -> str:
        self.calls.append(CallRecord(system_prompt=system_prompt, prompt=prompt, timeout=timeout))
        raise RuntimeError("text transport failed")


class VisionOnlyLLM:
    """Only has vision method (which fails). Tests tolerance for missing text method."""

    def complete_prompt_with_images(
        self,
        system_prompt: str,
        prompt: str,
        image_paths: list[Path],
        timeout: float | None = None,
    ) -> str:
        raise RuntimeError("vision transport failed")
