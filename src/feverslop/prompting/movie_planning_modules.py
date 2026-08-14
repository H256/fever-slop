from __future__ import annotations

import json
from typing import Any

from feverslop.prompting.guide_loader import load_markdown_guide
from feverslop.prompting.movie_planning_signatures import build_movie_planning_signature_bundle


def _value(result: Any, name: str) -> Any:
    if isinstance(result, dict):
        return result.get(name, result)
    return getattr(result, name, result)


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return value.__dict__
    if isinstance(value, tuple):
        return list(value)
    return str(value)


class MoviePlanningModules:
    """DSPy-backed Movie contracts with a legacy transport compatibility seam."""

    def __init__(self, llm: Any, *, dspy_runtime: Any | None = None):
        self._llm = llm
        self._predictors: dict[str, Any] = {}
        if isinstance(getattr(llm, "model", None), str) and getattr(llm, "client", None) is not None:
            import dspy

            runtime = dspy_runtime
            if runtime is None:
                from feverslop.prompting.dspy_runtime import DspyRuntime

                runtime = DspyRuntime.create(dspy)
            self._lm = runtime.make_lm(llm)
            self._context = runtime.context
            for name, signature in build_movie_planning_signature_bundle(dspy).items():
                self._predictors[name] = runtime.predict(signature)

    def _call(self, name: str, payload: dict[str, Any], output: str, *, timeout: float | None = None, system_prompt: str = "") -> Any:
        guide_name = {
            "movie_bible": "movie-bible",
            "shot_plan_from_bible": "movie-shot-plan-bible",
        }.get(name, f"movie-{name.replace('_', '-')}")
        guide = load_markdown_guide(guide_name)
        if not self._predictors:
            prompt = f"{guide}\n\nStructured input data:\n{json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)}"
            kwargs: dict[str, Any] = {"system_prompt": system_prompt or guide, "prompt": prompt}
            if timeout is not None:
                kwargs["timeout"] = timeout
            return self._llm.complete_prompt(**kwargs)
        kwargs: dict[str, Any] = {"guide": guide, "payload": payload}
        if timeout is not None:
            kwargs["config"] = {"timeout": timeout}
        with self._context(lm=self._lm):
            return _value(self._predictors[name](**kwargs), output)

    def story_arch(self, payload: dict[str, Any], *, timeout: float | None = None) -> Any:
        return self._call("story_arch", payload, "result", timeout=timeout, system_prompt="You are a film writer. Return ONLY valid JSON.")

    def movie_bible(self, payload: dict[str, Any], *, timeout: float | None = None) -> Any:
        return self._call("movie_bible", payload, "result", timeout=timeout, system_prompt="You are a film development producer. Return ONLY valid JSON.")

    def refine_locations(self, payload: dict[str, Any], *, timeout: float | None = None) -> Any:
        return self._call("refine_locations", payload, "result", timeout=timeout, system_prompt="You are a production designer. Return ONLY valid JSON.")

    def refine_actors(self, payload: dict[str, Any], *, timeout: float | None = None) -> Any:
        return self._call("refine_actors", payload, "result", timeout=timeout, system_prompt="You are a character designer. Return ONLY valid JSON.")

    def continuity_plan(self, payload: dict[str, Any], *, timeout: float | None = None) -> Any:
        return self._call("continuity_plan", payload, "result", timeout=timeout, system_prompt="You are a film continuity supervisor. Return ONLY valid JSON.")

    def story_design(self, payload: dict[str, Any], *, timeout: float | None = None) -> Any:
        return self._call("story_design", payload, "result", timeout=timeout, system_prompt="You are a dramaturg and story editor. Return ONLY valid JSON.")

    def screenplay(self, payload: dict[str, Any], *, timeout: float | None = None) -> Any:
        return self._call("screenplay", payload, "result", timeout=timeout, system_prompt="You are a film screenwriter. Return ONLY valid JSON.")

    def narrative_plan(self, payload: dict[str, Any], *, timeout: float | None = None) -> Any:
        return self._call("narrative_plan", payload, "result", timeout=timeout, system_prompt="You are a film story editor. Return ONLY valid JSON.")

    def shot_plan_from_bible(self, payload: dict[str, Any], *, timeout: float | None = None) -> Any:
        return self._call("shot_plan_from_bible", payload, "result", timeout=timeout, system_prompt="You are a film director and shot planner. Return ONLY valid JSON.")

    def shot_plan(self, payload: dict[str, Any], *, timeout: float | None = None) -> Any:
        return self._call("shot_plan", payload, "result", timeout=timeout, system_prompt="You are a film director and shot planner. Return ONLY valid JSON.")
