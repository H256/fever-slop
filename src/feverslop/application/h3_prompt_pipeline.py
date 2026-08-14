from __future__ import annotations

from typing import Any, Callable

from feverslop.application.pipeline_context import GenerateRenderPlanContext
from feverslop.ports.generate_pipeline import H3PromptBuilderFactory
from feverslop.prompting.dspy_h3_models import PromptMode
from feverslop.prompting.model_types import resolve_model_type
from feverslop.utils.sub_step_progress import SubStepProgress


def _attach_relay_segments(stage1_segments: list[dict], relay_scenes: list[dict]) -> list[dict]:
    """Join the frame relay artifact to the segment records consumed by H3."""
    relay_by_segment = {
        str(scene.get("metadata", {}).get("segment_id") or scene.get("segment_id")): scene
        for scene in relay_scenes
        if scene.get("metadata", {}).get("segment_id") or scene.get("segment_id")
    }
    enriched = []
    for segment in stage1_segments:
        result = dict(segment)
        relay_scene = relay_by_segment.get(str(segment.get("segment_id")))
        if relay_scene:
            result.setdefault("fps", relay_scene.get("fps"))
            result.setdefault("duration_seconds", relay_scene.get("duration_seconds"))
            if "ltx" not in result and relay_scene.get("ltx"):
                result["ltx"] = relay_scene["ltx"]
        enriched.append(result)
    return enriched


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

        relay_path = context.setdefault("ltx_prompt_relay_json", None)
        if relay_path is not None:
            stage1_segments = _attach_relay_segments(
                stage1_segments,
                artifact_store.read_json(relay_path),
            )

        log_step("8.5. H3 Structured Prompts")
        llm = self.llm_factory(app_config)
        builder_factory = self.h3_prompt_builder_factory
        try:
            model_spec = resolve_model_type(config.video_pipeline)
        except ValueError:
            model_spec = None
        if model_spec and model_spec.is_minimax_h3 and self.dspy_prompt_builder_factory:
            builder_factory = self.dspy_prompt_builder_factory
        builder = builder_factory(llm)

        mode = model_spec.prompt_mode.value if model_spec else PromptMode.T2V.value
        stem_files = context["stem_files"] if "stem_files" in context.keys() else None
        audio_paths = (
            _configured_audio_paths(config, stem_files, getattr(config, "input_audio", None))
            if model_spec and model_spec.prompt_mode is PromptMode.R2V
            else stem_files
        )

        reporter = context["reporter"] if "reporter" in context.keys() else None
        progress = SubStepProgress(reporter, "H3 prompts", len(stage1_segments))
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
            progress_callback=lambda current, total: progress.update(current),
            status_callback=lambda current, total, status: (
                reporter.message(
                    f"[cyan]H3 prompts: {current}/{total} scenes - "
                    f"{'start' if status == 'started' else 'completed'}[/cyan]"
                )
                if reporter is not None
                else None
            ),
        )
        log_file("H3 Prompts JSON", h3_prompts_json)
        context["h3_prompts"] = artifact_store.read_json(h3_prompts_json)
        return context
