from __future__ import annotations

from typing import Any

from feverslop.prompting.dspy_runtime import DspyRuntime
from feverslop.prompting.general_signatures import (
    LyricCorrections,
    PromptResult,
    SongBriefResult,
    StoryboardPromptResult,
    build_general_signature_bundle,
    parse_prompt_result,
)
from feverslop.prompting.guide_loader import load_markdown_guide
from feverslop.prompting.llm_policy import lyric_alignment_max_tokens, policy_for


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
        if not isinstance(getattr(llm, "model", None), str) or getattr(llm, "client", None) is None:
            raise RuntimeError("DSPy prompt modules require a configured DSPy-compatible LLM")
        import dspy

        runtime = dspy_runtime or DspyRuntime.create(dspy)
        self._lm = runtime.make_lm(llm)
        self._context = runtime.context
        self._predictors = {
            name: runtime.predict(signature)
            for name, signature in build_general_signature_bundle(dspy).items()
        }

    def _call(
        self,
        name: str,
        guide_name: str,
        payload: dict[str, Any],
        output_type: Any,
        *,
        timeout=None,
        max_tokens: int | None = None,
        **extra,
    ):
        guide = load_markdown_guide(guide_name)
        kwargs = {"guide": guide, **payload, **extra}
        config = {"max_tokens": max_tokens or policy_for(name).max_tokens}
        if timeout is not None:
            config["timeout"] = timeout
        kwargs["config"] = config
        with self._context(lm=self._lm):
            result = _value(self._predictors[name](**kwargs), "result")
        if output_type is PromptResult:
            return parse_prompt_result(result)
        return output_type.model_validate(result)

    def song_brief(self, request: dict[str, Any], *, timeout=None) -> SongBriefResult:
        return self._call("song_brief", "song-brief", {"request": request}, SongBriefResult, timeout=timeout)

    def lyric_alignment(self, request: dict[str, Any], *, timeout=None) -> LyricCorrections:
        segment_count = len(request.get("WHISPER_SEGMENTS", []))
        return self._call(
            "lyric_alignment",
            "lyric-alignment",
            {"request": request},
            LyricCorrections,
            timeout=timeout,
            max_tokens=lyric_alignment_max_tokens(segment_count),
        )

    def zimage_prompt(self, payload: dict[str, Any], *, timeout=None) -> PromptResult:
        return self._call("zimage_prompt", "music-video-t2i", {"payload": payload}, PromptResult, timeout=timeout)

    def i2v_prompt(self, payload: dict[str, Any], *, guide: str, timeout=None) -> PromptResult:
        kwargs = {"guide": guide, "payload": payload}
        config = {"max_tokens": policy_for("i2v_prompt").max_tokens}
        if timeout is not None:
            config["timeout"] = timeout
        kwargs["config"] = config
        with self._context(lm=self._lm):
            return parse_prompt_result(_value(self._predictors["i2v_prompt"](**kwargs), "result"))

    def storyboard_transform(self, payload: dict[str, Any], *, timeout=None) -> StoryboardPromptResult:
        return self._call(
            "storyboard_transform",
            "storyboard-transform",
            payload,
            StoryboardPromptResult,
            timeout=timeout,
        )
