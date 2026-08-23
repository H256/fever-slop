from __future__ import annotations

from typing import Any

from feverslop.prompting.guide_loader import load_markdown_guide
from feverslop.prompting.llm_policy import (
    CONCEPT_MAP,
    DETAIL,
    I2V,
    REPAIR_CONCEPTS,
    STORY_IDEA,
    STYLE_BLOCK,
    SUBJECT_LOCATIONS,
    SUMMARY,
    T2I,
    concept_batch_max_tokens,
    policy_for,
)
from feverslop.prompting.music_video_signatures import (
    build_music_video_signature_bundle,
)


def _value(result: Any, name: str) -> Any:
    value = getattr(result, name, None)
    if value is not None:
        return value
    if isinstance(result, dict):
        return result.get(name)
    return result


class MusicVideoPromptModules:
    """Typed DSPy request boundary with a small legacy-fake compatibility seam."""

    def __init__(self, llm: Any, *, dspy_runtime: Any | None = None):
        if not isinstance(getattr(llm, "model", None), str) or getattr(llm, "client", None) is None:
            raise RuntimeError("DSPy music-video prompts require a configured DSPy-compatible LLM")
        self._predictors: dict[str, Any] = {}
        if isinstance(getattr(llm, "model", None), str) and getattr(llm, "client", None) is not None:
            import dspy

            signatures = build_music_video_signature_bundle(dspy)
            runtime = dspy_runtime
            if runtime is None:
                from feverslop.prompting.dspy_runtime import DspyRuntime

                runtime = DspyRuntime.create(dspy)
            self._lm = runtime.make_lm(llm)
            self._context = runtime.context
            for name, signature in signatures.items():
                self._predictors[name] = runtime.predict(signature)

    def _call(
        self,
        name: str,
        guide: str,
        payload: dict[str, Any],
        output: str,
        *,
        timeout=None,
        max_tokens: int | None = None,
    ):
        predictor_kwargs = {"guide": guide, **payload}
        config = {"max_tokens": max_tokens or policy_for(name).max_tokens}
        if timeout is not None:
            config["timeout"] = timeout
        predictor_kwargs["config"] = config
        with self._context(lm=self._lm):
            return _value(self._predictors[name](**predictor_kwargs), output)

    def story_idea(self, lyrics: str, notes: str = "") -> str:
        return str(self._call(STORY_IDEA, load_markdown_guide("music-video-story-idea"), {"lyrics": lyrics, "notes": notes}, "story_idea")).strip()

    def style_block(self, lyrics: str, notes: str = "") -> str:
        return str(self._call(STYLE_BLOCK, load_markdown_guide("music-video-style"), {"lyrics": lyrics, "notes": notes}, "style_block")).strip()

    def subject_locations(self, story_idea: str, notes: str = "") -> Any:
        return self._call(SUBJECT_LOCATIONS, load_markdown_guide("music-video-subject-locations"), {"story_idea": story_idea, "notes": notes}, "result")

    def concepts(self, payload: dict[str, Any], *, batch: bool = False, silent_mode: bool = False, timeout=None) -> Any:
        guide = load_markdown_guide("music-video-concepts")
        batch_size = len(payload.get("CURRENT_BATCH_SEGMENTS", [])) if batch else 0
        if not batch:
            batch_size = len(payload.get("SEGMENT_TIMELINE_JSON", []))
        if batch:
            guide += "\n\nThis request contains one batch only; preserve continuity with prior progress."
        if silent_mode:
            guide += "\n\nSilent mode is active: do not create singing, lip-sync, vocal performance, mouth performance, or dialogue delivery."
        return self._call(
            CONCEPT_MAP,
            guide,
            {"payload": payload},
            "concepts",
            timeout=timeout,
            max_tokens=concept_batch_max_tokens(batch_size) if batch_size else None,
        )

    def repair_concepts(self, payload: dict[str, Any], *, timeout=None) -> Any:
        return self._call(REPAIR_CONCEPTS, load_markdown_guide("music-video-concept-repair"), {"payload": payload}, "concepts", timeout=timeout)

    def summary(self, payload: dict[str, Any], *, timeout=None) -> str:
        return str(self._call(SUMMARY, load_markdown_guide("music-video-summary"), {"payload": payload}, "summary", timeout=timeout)).strip()

    def detail(self, label: str, payload: dict[str, Any], guide: str, *, timeout=None) -> str:
        return str(self._call(DETAIL, guide, {"label": label, "payload": payload}, "detail", timeout=timeout)).strip()

    def t2i(self, payload: dict[str, Any], guide: str) -> str:
        return str(self._call(T2I, guide, {"payload": payload}, "prompt")).strip()

    def i2v(self, payload: dict[str, Any], guide: str, performance_policy: str) -> str:
        return str(self._call(I2V, guide, {"performance_policy": performance_policy, "payload": payload}, "prompt")).strip()
