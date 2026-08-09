from __future__ import annotations

from typing import Any, Callable

from feverslop.application.pipeline_context import GenerateRenderPlanContext
from feverslop.ports.generate_pipeline import H3PromptBuilderFactory


def _configured_audio_paths(
    config: Any,
    stem_files: dict[str, Any] | None,
    input_audio: Any | None = None,
) -> dict[str, Any] | None:
    """Return only the audio stems selected for the MiniMax reference workflow."""
    if not stem_files and input_audio is None:
        return None

    configured_stems = list(getattr(getattr(config, "minimax_h3_audio_refs", None), "stems", ()))
    if not configured_stems:
        return None

    available = dict(stem_files or {})
    if input_audio is not None:
        available.setdefault("full_mix", input_audio)
    selected = {
        stem_name: available[stem_name]
        for stem_name in configured_stems
        if stem_name in available
    }
    return selected or None


class H3PromptPipeline:
    """Application service for H3-structured prompt generation (stage 8.5)."""

    defer_until_references = True

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
        audio_paths = (
            _configured_audio_paths(config, stem_files, getattr(config, "input_audio", None))
            if config.video_pipeline == "minimax-h3-r2v"
            else stem_files
        )

        builder.build_all_h3_prompts(
            stage1_segments=stage1_segments,
            concept_prompts=concept_prompts,
            scene_details=scene_details,
            global_context=global_context,
            mode=mode,
            video_type="music_video",
            output_json_path=h3_prompts_json,
            artifact_store=artifact_store,
            audio_paths=audio_paths,
            reference_root=getattr(config, "project_dir", None),
            progress_callback=lambda current, total: context["reporter"].message(
                f"[cyan]H3 prompts: {current}/{total} scenes[/cyan]"
            ),
            status_callback=lambda current, total, status: context["reporter"].message(
                f"[cyan]H3 prompts: {current}/{total} scenes - "
                f"{'start' if status == 'started' else 'completed'}[/cyan]"
            ),
        )
        log_file("H3 Prompts JSON", h3_prompts_json)
        context["h3_prompts"] = artifact_store.read_json(h3_prompts_json)
        return context
