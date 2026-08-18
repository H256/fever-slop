from __future__ import annotations

from pathlib import Path
from typing import Any

from feverslop.prompting.dspy_runtime import DspyRuntime
from feverslop.prompting.guide_loader import load_markdown_guide
from feverslop.prompting.llm_policy import msr_segments_max_tokens
from feverslop.prompting.msr_signatures import MSRPromptResult, build_msr_signature_bundle


def _value(result: Any, name: str) -> Any:
    value = getattr(result, name, None)
    if value is not None:
        return value
    if isinstance(result, dict):
        return result.get(name)
    return result


class MSRPromptModules:
    """DSPy boundary for MSR vision and segment prompt contracts."""

    def __init__(self, llm: Any, *, dspy_runtime: Any | None = None):
        if not isinstance(getattr(llm, "model", None), str) or getattr(llm, "client", None) is None:
            raise RuntimeError("DSPy MSR prompts require a configured DSPy-compatible LLM")
        if dspy_runtime is None:
            import dspy

            dspy_runtime = DspyRuntime.create(dspy)
        else:
            dspy = __import__("dspy")
        self._lm = dspy_runtime.make_lm(llm)
        self._context = dspy_runtime.context
        signatures = build_msr_signature_bundle(dspy)
        self._predictors = {name: dspy_runtime.predict(signature) for name, signature in signatures.items()}

    def _call(
        self,
        name: str,
        guide_name: str,
        payload: dict[str, Any],
        *,
        images: list[Any] | None = None,
        timeout: float | None = None,
        max_tokens: int | None = None,
    ) -> MSRPromptResult:
        kwargs: dict[str, Any] = {
            "guide": load_markdown_guide(guide_name),
            "payload": payload,
        }
        if images is not None:
            kwargs["images"] = images
        if timeout is not None:
            kwargs["config"] = {"timeout": timeout}
        if max_tokens is not None:
            kwargs.setdefault("config", {})["max_tokens"] = max_tokens
        with self._context(lm=self._lm):
            result = _value(self._predictors[name](**kwargs), "result")
        return MSRPromptResult.model_validate(result)

    def vision(self, payload: dict[str, Any], images: list[Path], *, timeout: float | None = None) -> MSRPromptResult:
        import dspy

        return self._call(
            "vision",
            "msr-vision",
            payload,
            images=[image if isinstance(image, dspy.Image) else dspy.Image.from_path(str(image)) for image in images],
            timeout=timeout,
        )

    def segments(self, payload: dict[str, Any], *, timeout: float | None = None) -> MSRPromptResult:
        relay_count = len(payload.get("relay_segments", []))
        return self._call(
            "segments",
            "msr-segments",
            payload,
            timeout=timeout,
            max_tokens=msr_segments_max_tokens(relay_count),
        )
