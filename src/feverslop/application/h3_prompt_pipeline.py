from __future__ import annotations

from typing import Any, Callable

from feverslop.application.pipeline_context import GenerateRenderPlanContext
from feverslop.ports.generate_pipeline import H3PromptBuilderFactory


class H3PromptPipeline:
    """Application service for H3-structured prompt generation (stage 8.5)."""

    required_keys = {
        "scene_prompts_json",
        "stage1_segments",
        "concept_prompts",
        "scene_details",
        "global_context",
        "h3_prompts_json",
        "app_config",
    }
    produced_keys = {"h3_prompts"}

    def __init__(
        self,
        *,
        llm_factory: Callable[[Any], Any],
        h3_prompt_builder_factory: H3PromptBuilderFactory,
    ):
        self.llm_factory = llm_factory
        self.h3_prompt_builder_factory = h3_prompt_builder_factory

    def execute(self, context: GenerateRenderPlanContext) -> GenerateRenderPlanContext:
        missing = self.required_keys - context.keys()
        if missing:
            raise KeyError(f"{self.__class__.__name__} missing context keys: {sorted(missing)}")
        return self.run(context)

    def run(self, context: GenerateRenderPlanContext) -> GenerateRenderPlanContext:
        app_config = context["app_config"]
        stage1_segments = context["stage1_segments"]
        concept_prompts = context["concept_prompts"]
        scene_details = context["scene_details"]
        global_context = context["global_context"]
        h3_prompts_json = context["h3_prompts_json"]
        artifact_store = context["artifact_store"]
        log_step = context["log_step"]
        log_file = context["log_file"]
        run_spinner = context["run_spinner"]

        log_step("8.5. H3 Structured Prompts")
        llm = self.llm_factory(app_config)
        builder = self.h3_prompt_builder_factory(llm)

        run_spinner(
            "Generating H3-structured prompts per scene...",
            lambda: builder.build_all_h3_prompts(
                stage1_segments=stage1_segments,
                concept_prompts=concept_prompts,
                scene_details=scene_details,
                global_context=global_context,
                mode="base",
                video_type="music_video",
                output_json_path=h3_prompts_json,
                artifact_store=artifact_store,
            ),
        )
        log_file("H3 Prompts JSON", h3_prompts_json)
        context["h3_prompts"] = artifact_store.read_json(h3_prompts_json)
        return context
