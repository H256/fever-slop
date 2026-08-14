from __future__ import annotations

from typing import Any

from feverslop.prompting.guide_loader import load_markdown_guide
from feverslop.prompting.movie_planning_signatures import build_movie_planning_signature_bundle
from feverslop.prompting.movie_planning_signatures import (
    ContinuityPlanPayload,
    MovieBiblePayload,
    NarrativePlanPayload,
    RefineActorsPayload,
    RefineLocationsPayload,
    ScreenplayPayload,
    ShotPlanFromBiblePayload,
    ShotPlanPayload,
    StoryArchPayload,
    StoryDesignPayload,
)


_PAYLOAD_TYPES = {
    "story_arch": StoryArchPayload,
    "movie_bible": MovieBiblePayload,
    "refine_locations": RefineLocationsPayload,
    "refine_actors": RefineActorsPayload,
    "continuity_plan": ContinuityPlanPayload,
    "story_design": StoryDesignPayload,
    "screenplay": ScreenplayPayload,
    "narrative_plan": NarrativePlanPayload,
    "shot_plan_from_bible": ShotPlanFromBiblePayload,
    "shot_plan": ShotPlanPayload,
}


def _value(result: Any, name: str) -> Any:
    if isinstance(result, dict):
        return result.get(name, result)
    return getattr(result, name, result)


class MoviePlanningModules:
    """DSPy-backed Movie planning contracts."""

    def __init__(self, llm: Any, *, dspy_runtime: Any | None = None):
        self._llm = llm
        self._predictors: dict[str, Any] = {}
        if not isinstance(getattr(llm, "model", None), str) or getattr(llm, "client", None) is None:
            raise RuntimeError("DSPy movie planning requires a configured DSPy-compatible LLM")
        runtime = dspy_runtime
        if runtime is None:
            try:
                import dspy
            except ImportError as exc:
                raise RuntimeError("DSPy is required for movie planning; install the dspy dependency") from exc
            from feverslop.prompting.dspy_runtime import DspyRuntime

            runtime = DspyRuntime.create(dspy)
        else:
            dspy = __import__("dspy")
        self._lm = runtime.make_lm(llm)
        self._context = runtime.context
        self._predictors = {
            name: runtime.predict(signature)
            for name, signature in build_movie_planning_signature_bundle(dspy).items()
        }

    def _call(self, name: str, payload: dict[str, Any], output: str, *, timeout: float | None = None, system_prompt: str = "") -> Any:
        guide_name = {
            "movie_bible": "movie-bible",
            "shot_plan_from_bible": "movie-shot-plan-bible",
        }.get(name, f"movie-{name.replace('_', '-')}")
        guide = load_markdown_guide(guide_name)
        payload = _PAYLOAD_TYPES[name].model_validate(payload)
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
