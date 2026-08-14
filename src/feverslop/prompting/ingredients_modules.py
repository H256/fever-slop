from __future__ import annotations

from pathlib import Path
from typing import Any

from feverslop.prompting.dspy_runtime import DspyRuntime
from feverslop.prompting.guide_loader import load_markdown_guide
from feverslop.prompting.ingredients_signatures import (
    IngredientsVisionPayload,
    IngredientsVisionResult,
    build_ingredients_signature_bundle,
)


def _value(result: Any, name: str) -> Any:
    if isinstance(result, dict):
        return result.get(name, result)
    return getattr(result, name, result)


class IngredientsPromptModules:
    """DSPy boundary for the multimodal Ingredients vision contract."""

    def __init__(self, llm: Any, *, dspy_runtime: Any | None = None, image_factory=None):
        if not isinstance(getattr(llm, "model", None), str) or getattr(llm, "client", None) is None:
            raise RuntimeError("DSPy Ingredients prompts require a configured DSPy-compatible LLM")
        runtime = dspy_runtime
        if runtime is None:
            try:
                import dspy
            except ImportError as exc:
                raise RuntimeError("DSPy is required for Ingredients prompts; install the dspy dependency") from exc
            runtime = DspyRuntime.create(dspy)
        else:
            dspy = __import__("dspy")
        self._lm = runtime.make_lm(llm)
        self._context = runtime.context
        self._image_factory = image_factory
        self._predictor = runtime.predict(build_ingredients_signature_bundle(dspy)["vision"])

    def vision(
        self,
        payload: dict[str, Any],
        images: list[Path],
        *,
        timeout: float | None = None,
    ) -> IngredientsVisionResult:
        kwargs: dict[str, Any] = {
            "guide": load_markdown_guide("ingredients-vision"),
            "payload": IngredientsVisionPayload.model_validate(payload),
            "images": [self._image(path) for path in images],
        }
        if timeout is not None:
            kwargs["config"] = {"timeout": timeout}
        with self._context(lm=self._lm):
            result = _value(self._predictor(**kwargs), "result")
        return IngredientsVisionResult.model_validate(result)

    def _image(self, path: Path) -> Any:
        if self._image_factory is not None:
            return self._image_factory(path)
        import dspy
        return dspy.Image.from_path(str(path))
