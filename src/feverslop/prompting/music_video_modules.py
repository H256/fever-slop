from __future__ import annotations

import json
from typing import Any

from feverslop.prompting.guide_loader import load_markdown_guide
from feverslop.prompting.music_video_signatures import build_music_video_signature_bundle
from feverslop.prompting.dspy_runtime import DspyRuntime


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
        self._llm = llm
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

    def _call(self, name: str, guide: str, payload: dict[str, Any], output: str, *, timeout=None):
        if not self._predictors:
            kwargs = {"system_prompt": guide, "prompt": json.dumps(payload.get("payload", payload), ensure_ascii=False, indent=2)}
            if timeout is not None:
                kwargs["timeout"] = timeout
            return DspyRuntime.complete_text(self._llm, **kwargs)
        predictor_kwargs = {"guide": guide, **payload}
        if timeout is not None:
            predictor_kwargs["config"] = {"timeout": timeout}
        with self._context(lm=self._lm):
            return _value(self._predictors[name](**predictor_kwargs), output)

    def story_idea(self, lyrics: str, notes: str = "") -> str:
        return str(self._call("story_idea", load_markdown_guide("music-video-story-idea"), {"lyrics": lyrics, "notes": notes}, "story_idea")).strip()

    def style_block(self, lyrics: str, notes: str = "") -> str:
        return str(self._call("style_block", load_markdown_guide("music-video-style"), {"lyrics": lyrics, "notes": notes}, "style_block")).strip()

    def subject_locations(self, story_idea: str, notes: str = "") -> Any:
        return self._call("subject_locations", load_markdown_guide("music-video-subject-locations"), {"story_idea": story_idea, "notes": notes}, "result")

    def concepts(self, payload: dict[str, Any], *, batch: bool = False, silent_mode: bool = False, timeout=None) -> Any:
        guide = load_markdown_guide("music-video-concepts")
        if batch:
            guide += "\n\nThis request contains one batch only; preserve continuity with prior progress."
        if silent_mode:
            guide += "\n\nSilent mode is active: do not create singing, lip-sync, vocal performance, mouth performance, or dialogue delivery."
        return self._call("concept_map", guide, {"payload": payload}, "concepts", timeout=timeout)

    def repair_concepts(self, payload: dict[str, Any], *, timeout=None) -> Any:
        return self._call("repair_concepts", load_markdown_guide("music-video-concept-repair"), {"payload": payload}, "concepts", timeout=timeout)

    def summary(self, payload: dict[str, Any], *, timeout=None) -> str:
        return str(self._call("summary", load_markdown_guide("music-video-summary"), {"payload": payload}, "summary", timeout=timeout)).strip()

    def detail(self, label: str, payload: dict[str, Any], guide: str, *, timeout=None) -> str:
        return str(self._call("detail", guide, {"label": label, "payload": payload}, "detail", timeout=timeout)).strip()

    def t2i(self, payload: dict[str, Any], guide: str) -> str:
        return str(self._call("t2i", guide, {"payload": payload}, "prompt")).strip()

    def i2v(self, payload: dict[str, Any], guide: str, performance_policy: str) -> str:
        return str(self._call("i2v", guide, {"performance_policy": performance_policy, "payload": payload}, "prompt")).strip()
