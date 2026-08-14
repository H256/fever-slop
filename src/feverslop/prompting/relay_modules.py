from __future__ import annotations

from typing import Any

from feverslop.prompting.dspy_runtime import DspyRuntime
from feverslop.prompting.guide_loader import load_markdown_guide
from feverslop.prompting.relay_signatures import RelayDirectionResult, build_relay_signature_bundle


def _value(result: Any, name: str) -> Any:
    value = getattr(result, name, None)
    if value is not None:
        return value
    if isinstance(result, dict):
        return result.get(name)
    return result


class RelayPromptModules:
    def __init__(self, llm: Any, *, dspy_runtime: Any | None = None):
        if not isinstance(getattr(llm, "model", None), str) or getattr(llm, "client", None) is None:
            raise RuntimeError("DSPy relay prompts require a configured DSPy-compatible LLM")
        if dspy_runtime is None:
            import dspy

            dspy_runtime = DspyRuntime.create(dspy)
        else:
            dspy = __import__("dspy")
        self._lm = dspy_runtime.make_lm(llm)
        self._context = dspy_runtime.context
        self._predictor = dspy_runtime.predict(build_relay_signature_bundle(dspy))

    def compact(self, payload: dict[str, Any], *, max_words: int, subject_anchor: str, timeout: float | None = None) -> RelayDirectionResult:
        kwargs: dict[str, Any] = {
            "guide": load_markdown_guide("relay-directions"),
            "payload": payload,
            "max_words": max_words,
            "subject_anchor": subject_anchor,
        }
        if timeout is not None:
            kwargs["config"] = {"timeout": timeout}
        with self._context(lm=self._lm):
            result = _value(self._predictor(**kwargs), "result")
        return RelayDirectionResult.model_validate(result)
