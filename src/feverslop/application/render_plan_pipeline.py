from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from feverslop.application.pipeline_context import GenerateRenderPlanContext


class RenderPlanPipeline:
    """Application service for final render plan assembly."""

    required_keys = {"scene_prompts_json", "ltx_prompt_relay_json", "render_plan_json", "video_settings"}
    produced_keys = {"render_plan"}

    def __init__(self, *, build_render_plan: Callable[..., Any]):
        self.build_render_plan = build_render_plan

    def execute(self, context: GenerateRenderPlanContext) -> GenerateRenderPlanContext:
        missing = self.required_keys - context.keys()
        if missing:
            raise KeyError(f"{self.__class__.__name__} missing context keys: {sorted(missing)}")
        return self.run(context)

    def run(self, context: GenerateRenderPlanContext) -> GenerateRenderPlanContext:
        render_plan_json = context["render_plan_json"]
        artifact_store = context["artifact_store"]
        log_step = context["log_step"]
        log_file = context["log_file"]

        # -- stem files (MiniMax H3 R2V) --
        config = context.get("config", None)
        stem_list: list[str] | None = None
        input_audio: Path | None = None
        stem_files: dict[str, Path] | None = None

        if context.get("stem_files") is not None:
            stem_files = context["stem_files"]
            if config is not None:
                stem_list = list(config.minimax_h3_audio_refs.stems)
                input_audio = config.input_audio

        log_step("9. Render Plan")
        self.build_render_plan(
            scene_prompts_json=context["scene_prompts_json"],
            ltx_prompt_relay_json=context["ltx_prompt_relay_json"],
            output_json_file=render_plan_json,
            video_settings=context["video_settings"],
            artifact_store=artifact_store,
            h3_prompts_json=context["h3_prompts_json"],
            stem_list=stem_list,
            input_audio=input_audio,
            stem_files=stem_files,
        )
        log_file("Render Plan JSON", render_plan_json)
        context["render_plan"] = artifact_store.read_json(render_plan_json)
        return context
