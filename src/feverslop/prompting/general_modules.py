from __future__ import annotations

import json
from typing import Any

from feverslop.prompting.dspy_runtime import DspyRuntime
from feverslop.prompting.general_signatures import (
    LyricCorrections,
    PromptResult,
    SongBriefResult,
    build_general_signature_bundle,
)
from feverslop.prompting.guide_loader import load_markdown_guide


def _value(result: Any, name: str) -> Any:
    value = getattr(result, name, None)
    if value is not None:
        return value
    if isinstance(result, dict):
        return result.get(name)
    return result


class GeneralPromptModules:
    """Typed DSPy contracts for the remaining standalone prompt requests."""

    def __init__(self, llm: Any, *, dspy_runtime: Any | None = None):
        self._llm = llm
        self._runtime = dspy_runtime
        self._predictors: dict[str, Any] = {}
        if isinstance(getattr(llm, "model", None), str) and getattr(llm, "client", None) is not None:
            import dspy

            runtime = dspy_runtime or DspyRuntime.create(dspy)
            self._runtime = runtime
            self._lm = runtime.make_lm(llm)
            self._context = runtime.context
            self._predictors = {
                name: runtime.predict(signature)
                for name, signature in build_general_signature_bundle(dspy).items()
            }

    def _call(self, name: str, guide_name: str, payload: dict[str, Any], output_type: Any, *, timeout=None, legacy_system_prompt=None, legacy_payload=None, **extra):
        guide = load_markdown_guide(guide_name)
        if not self._predictors:
            response = DspyRuntime.complete_text(
                self._llm,
                system_prompt=legacy_system_prompt or guide,
                prompt=json.dumps(legacy_payload if legacy_payload is not None else payload, ensure_ascii=False, indent=2),
                timeout=timeout,
            )
            return response
        kwargs = {"guide": guide, **payload, **extra}
        if timeout is not None:
            kwargs["config"] = {"timeout": timeout}
        with self._context(lm=self._lm):
            result = _value(self._predictors[name](**kwargs), "result")
        return output_type.model_validate(result)

    def song_brief(self, request: dict[str, Any], *, timeout=None, legacy_system_prompt=None) -> SongBriefResult | str:
        return self._call("song_brief", "song-brief", {"request": request}, SongBriefResult, timeout=timeout, legacy_system_prompt=legacy_system_prompt, legacy_payload=request)

    def lyric_alignment(self, request: dict[str, Any], *, timeout=None, legacy_system_prompt=None) -> LyricCorrections | str:
        return self._call("lyric_alignment", "lyric-alignment", {"request": request}, LyricCorrections, timeout=timeout, legacy_system_prompt=legacy_system_prompt, legacy_payload=request)

    def zimage_prompt(self, payload: dict[str, Any], *, timeout=None, legacy_system_prompt=None) -> PromptResult | str:
        return self._call("zimage_prompt", "music-video-t2i", {"payload": payload}, PromptResult, timeout=timeout, legacy_system_prompt=legacy_system_prompt, legacy_payload=payload)

    def i2v_prompt(self, payload: dict[str, Any], *, guide: str, timeout=None) -> PromptResult | str:
        if self._predictors:
            kwargs = {"guide": guide, "payload": payload}
            if timeout is not None:
                kwargs["config"] = {"timeout": timeout}
            with self._context(lm=self._lm):
                return PromptResult.model_validate(_value(self._predictors["i2v_prompt"](**kwargs), "result"))
        return DspyRuntime.complete_text(self._llm, system_prompt=guide, prompt=json.dumps(payload, ensure_ascii=False, indent=2), timeout=timeout)

    def storyboard_transform(self, payload: dict[str, Any], *, timeout=None, legacy_system_prompt=None, legacy_prompt=None) -> PromptResult | str:
        if not self._predictors:
            return DspyRuntime.complete_text(self._llm, system_prompt=legacy_system_prompt or payload["system_template"], prompt=legacy_prompt or payload["user_template"], timeout=timeout)
        return self._call("storyboard_transform", "storyboard-transform", payload, PromptResult, timeout=timeout, legacy_system_prompt=legacy_system_prompt)
