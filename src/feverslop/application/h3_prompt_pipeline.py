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
        "config",
    }
    produced_keys = {"h3_prompts"}

    def __init__(
        self,
        *,
        llm_factory: Callable[[Any], Any],
        h3_prompt_builder_factory: H3PromptBuilderFactory,
        dspy_prompt_builder_factory: H3PromptBuilderFactory | None = None,
    ):
        self.llm_factory = llm_factory
        self.h3_prompt_builder_factory = h3_prompt_builder_factory
        self.dspy_prompt_builder_factory = dspy_prompt_builder_factory

    def execute(self, context: GenerateRenderPlanContext) -> GenerateRenderPlanContext:
        missing = self.required_keys - context.keys()
        if missing:
            raise KeyError(f"{self.__class__.__name__} missing context keys: {sorted(missing)}")
        return self.run(context)

    def run(self, context: GenerateRenderPlanContext) -> GenerateRenderPlanContext:
        app_config = context["app_config"]
        config = context["config"]
        stage1_segments = context["stage1_segments"]
        concept_prompts = context["concept_prompts"]
        scene_details = context["scene_details"]
        global_context = context["global_context"]
        h3_prompts_json = context["h3_prompts_json"]
        artifact_store = context["artifact_store"]
        log_step = context["log_step"]
        log_file = context["log_file"]

        log_step("8.5. H3 Structured Prompts")
        llm = self.llm_factory(app_config)
        builder_factory = self.h3_prompt_builder_factory
        if config.video_pipeline == "minimax-h3-r2v" and self.dspy_prompt_builder_factory:
            builder_factory = self.dspy_prompt_builder_factory
        builder = builder_factory(llm)

        if config.video_pipeline == "minimax-h3-r2v":
            mode = "ref"
        else:
            mode = "base"
        stem_files = context["stem_files"] if "stem_files" in context.keys() else None

        builder.build_all_h3_prompts(
            stage1_segments=stage1_segments,
            concept_prompts=concept_prompts,
            scene_details=scene_details,
            global_context=global_context,
            mode=mode,
            video_type="music_video",
            output_json_path=h3_prompts_json,
            artifact_store=artifact_store,
            audio_paths=stem_files,
            reference_root=getattr(config, "project_dir", None),
            progress_callback=lambda current, total: context["reporter"].message(
                f"[cyan]H3 prompts: {current}/{total} scenes[/cyan]"
            ),
        )
        log_file("H3 Prompts JSON", h3_prompts_json)
        context["h3_prompts"] = artifact_store.read_json(h3_prompts_json)
        return context
