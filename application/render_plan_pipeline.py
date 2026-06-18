from __future__ import annotations

from typing import Any

from render_plan_builder import build_render_plan


class RenderPlanPipeline:
    """Application service for final render plan assembly."""

    required_keys = {"scene_prompts_json", "ltx_prompt_relay_json", "render_plan_json", "video_settings"}
    produced_keys = {"render_plan"}

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        missing = self.required_keys - context.keys()
        if missing:
            raise KeyError(f"{self.__class__.__name__} missing context keys: {sorted(missing)}")
        return self.run(context)

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        render_plan_json = context["render_plan_json"]
        artifact_store = context["artifact_store"]
        log_step = context["log_step"]
        log_file = context["log_file"]

        log_step("9. Render Plan")
        build_render_plan(
            scene_prompts_json=context["scene_prompts_json"],
            ltx_prompt_relay_json=context["ltx_prompt_relay_json"],
            output_json_file=render_plan_json,
            video_settings=context["video_settings"],
            artifact_store=artifact_store,
        )
        log_file("Render Plan JSON", render_plan_json)
        context["render_plan"] = artifact_store.read_json(render_plan_json)
        return context
