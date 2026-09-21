from __future__ import annotations

from typing import Any

from feverslop.prompting.guide_loader import load_markdown_guide
from feverslop.prompting.llm_policy import (
    STORY_PLAN_ACTING,
    STORY_PLAN_BEAT_ALLOCATION,
    STORY_PLAN_BIBLE,
    STORY_PLAN_REPAIR,
    policy_for,
)
from feverslop.prompting.story_plan_signatures import (
    build_story_plan_signature_bundle,
)


_BUNDLE_TASK_NAMES = {
    STORY_PLAN_ACTING: "acting",
    STORY_PLAN_BEAT_ALLOCATION: "beat_allocation",
    STORY_PLAN_BIBLE: "bible",
    STORY_PLAN_REPAIR: "repair",
}


def _value(result: Any, name: str) -> Any:
    value = getattr(result, name, None)
    if value is not None:
        return value
    if isinstance(result, dict):
        return result.get(name)
    return result


class StoryPlanPromptModules:
    """Typed DSPy request boundary for the four story-plan jobs.

    One LM per task so each job resolves its per-task temperature
    (``DEFAULT_TASK_TEMPERATURES``) independently; inputs are compacted so
    acoustic evidence never reaches the model.
    """

    def __init__(self, llm: Any, *, dspy_runtime: Any | None = None):
        if not isinstance(getattr(llm, "model", None), str) or getattr(llm, "client", None) is None:
            raise RuntimeError("DSPy story-plan prompts require a configured DSPy-compatible LLM")
        import dspy

        bundle = build_story_plan_signature_bundle(dspy)
        runtime = dspy_runtime
        if runtime is None:
            from feverslop.prompting.dspy_runtime import DspyRuntime

            runtime = DspyRuntime.create(dspy)
        self._lms = {
            policy_name: runtime.make_lm(llm, task=policy_name)
            for policy_name in _BUNDLE_TASK_NAMES
        }
        self._context = runtime.context
        self._predictors = {
            policy_name: runtime.predict(bundle[_BUNDLE_TASK_NAMES[policy_name]])
            for policy_name in _BUNDLE_TASK_NAMES
        }

    def _call(
        self,
        name: str,
        guide: str,
        inputs: dict[str, Any],
        output: str,
        *,
        timeout: float | None = None,
    ) -> Any:
        from feverslop.prompting.planning_payload import compact_planning_payload

        predictor_kwargs = {key: compact_planning_payload(value) for key, value in inputs.items()}
        predictor_kwargs["guide"] = compact_planning_payload(guide)
        config = {"max_tokens": policy_for(name).max_tokens}
        if timeout is not None:
            config["timeout"] = timeout
        predictor_kwargs["config"] = config
        with self._context(lm=self._lms[name]):
            return _value(self._predictors[name](**predictor_kwargs), output)

    def bible(
        self,
        *,
        story_text: str,
        creative_direction: str,
        characters: list[dict[str, Any]],
        locations: list[dict[str, Any]],
        props: list[dict[str, Any]],
    ) -> Any:
        return self._call(
            STORY_PLAN_BIBLE,
            load_markdown_guide("story-plan-bible"),
            {
                "story_text": story_text,
                "creative_direction": creative_direction,
                "characters": characters,
                "locations": locations,
                "props": props,
            },
            "bible",
        )

    def beat_allocation(
        self,
        *,
        creative_direction: str,
        bible: Any,
        characters: list[dict[str, Any]],
        locations: list[dict[str, Any]],
        props: list[dict[str, Any]],
        segments: list[dict[str, Any]],
    ) -> Any:
        return self._call(
            STORY_PLAN_BEAT_ALLOCATION,
            load_markdown_guide("story-plan-beat-allocation"),
            {
                "creative_direction": creative_direction,
                "bible": bible,
                "characters": characters,
                "locations": locations,
                "props": props,
                "segments": segments,
            },
            "allocation",
        )

    def acting(
        self,
        *,
        creative_direction: str,
        bible: Any,
        beats: list[dict[str, Any]],
        segments: list[dict[str, Any]],
        characters: list[dict[str, Any]],
        locations: list[dict[str, Any]],
        props: list[dict[str, Any]],
    ) -> Any:
        return self._call(
            STORY_PLAN_ACTING,
            load_markdown_guide("story-plan-acting"),
            {
                "creative_direction": creative_direction,
                "bible": bible,
                "beats": beats,
                "segments": segments,
                "characters": characters,
                "locations": locations,
                "props": props,
            },
            "result",
        )

    def repair(
        self,
        *,
        prior_plan: dict[str, Any],
        diagnostics: list[dict[str, Any]],
    ) -> Any:
        return self._call(
            STORY_PLAN_REPAIR,
            load_markdown_guide("story-plan-repair"),
            {
                "prior_plan": prior_plan,
                "diagnostics": diagnostics,
            },
            "plan",
        )
