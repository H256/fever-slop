from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable

from feverslop.llm_concurrency import limit_dspy_lm


@dataclass(frozen=True, slots=True)
class H3SignatureBundle:
    analyze_image: Any
    build_prompt_plan: Any
    render_base_prompt: Any
    render_reference_prompt: Any


@dataclass(frozen=True, slots=True)
class DspyRuntime:
    """Small seam around DSPy factories so tests can inject fakes."""

    signatures: H3SignatureBundle
    lm_factory: Callable[..., Any]
    predict_factory: Callable[[Any], Any]
    context_factory: Callable[..., AbstractContextManager[Any]]

    @classmethod
    def create(cls, dspy_module: Any | None = None) -> "DspyRuntime":
        if dspy_module is None:
            import dspy as dspy_module

        from feverslop.prompting.dspy_h3_signatures import build_h3_signature_bundle

        return cls(
            signatures=build_h3_signature_bundle(dspy_module),
            lm_factory=dspy_module.LM,
            predict_factory=dspy_module.Predict,
            context_factory=dspy_module.context,
        )

    def predict(self, signature: Any) -> Any:
        return self.predict_factory(signature)

    def context(self, *, lm: Any) -> AbstractContextManager[Any]:
        return self.context_factory(lm=lm)

    def make_lm(self, llm: Any) -> Any:
        client = getattr(llm, "client", None)
        api_base = getattr(client, "base_url", None)
        if api_base is not None and not isinstance(api_base, str):
            api_base = str(api_base)
        kwargs = {
            "api_base": api_base,
            "api_key": getattr(client, "api_key", None),
            "temperature": getattr(llm, "dspy_temperature", 0.4),
            "max_tokens": llm.max_tokens,
            "cache": getattr(llm, "dspy_cache", False),
        }
        timeout = getattr(llm, "request_timeout_seconds", None)
        if timeout is not None:
            kwargs["timeout"] = timeout
        lm = self.lm_factory(f"openai/{llm.model}", **kwargs)
        limiter = getattr(llm, "llm_limiter", None)
        if limiter is None:
            return lm
        return limit_dspy_lm(lm, limiter)

    @staticmethod
    def complete_text(llm: Any, *, system_prompt: str, prompt: str, timeout: float | None = None) -> str:
        """Compatibility transport for non-DSPy test doubles and old clients."""
        return llm.complete_prompt(system_prompt=system_prompt, prompt=prompt, timeout=timeout)

    @staticmethod
    def complete_images(
        llm: Any,
        *,
        system_prompt: str,
        prompt: str,
        image_paths: list[Any],
        timeout: float | None = None,
    ) -> str:
        """Compatibility transport for non-DSPy vision test doubles."""
        try:
            return llm.complete_prompt_with_images(system_prompt, prompt, image_paths, timeout=timeout)
        except TypeError as exc:
            if "timeout" not in str(exc):
                raise
            return llm.complete_prompt_with_images(system_prompt, prompt, image_paths)
