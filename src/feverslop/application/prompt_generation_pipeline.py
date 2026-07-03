from __future__ import annotations

from pathlib import Path
import inspect
from typing import Any, Callable
from dataclasses import asdict, is_dataclass

from feverslop.application.pipeline_context import GenerateRenderPlanContext
from feverslop.ports.generate_pipeline import (
    ConceptBatcherFactory,
    LLMFactory,
    PromptPipelineFactory,
    ScenePromptBuilderFactory,
)
from feverslop.domain.prompt_constraints import build_location_constraint


def join_notes(*parts: str) -> str:
    return "\n".join(part.strip() for part in parts if part and part.strip())


def get_steering_value(config: Any, name: str, default: str = "") -> str:
    steering = getattr(config, "steering", None)
    return str(getattr(steering, name, default) or "")


def get_config_value(config: Any, name: str, default: Any = None) -> Any:
    return getattr(config, name, default)


def config_items_as_dicts(items: Any) -> list[dict]:
    output = []
    for item in items or []:
        if is_dataclass(item):
            output.append(asdict(item))
        elif isinstance(item, dict):
            output.append(dict(item))
    return output


def normalize_location_names(items: Any) -> list[str]:
    names = []
    for item in items or []:
        if isinstance(item, dict):
            names.append(str(item.get("name") or item.get("id") or "").strip())
        else:
            names.append(str(item).strip())
    return [name for name in names if name]


def reporter_message(reporter: Any, message: str) -> None:
    if reporter is not None:
        reporter.message(message)


def call_with_supported_kwargs(func: Callable[..., Any], **kwargs):
    signature = inspect.signature(func)
    supported = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return func(**supported)


def resolve_text_override(
    *,
    configured_value: Any,
    generated_value_factory: Callable[[], str],
    reporter: Any,
    message: str,
) -> str:
    configured = str(configured_value or "").strip()
    if configured:
        reporter_message(reporter, message)
        return configured
    return generated_value_factory()


def resolve_locations_override(*, configured_locations: Any, generated_locations: Any, reporter: Any) -> list[str]:
    if configured_locations:
        reporter_message(reporter, "[yellow]Using locations override from project config.[/yellow]")
        return configured_locations
    return normalize_location_names(generated_locations)


def validate_and_order_concept_prompts(stage1_segments: list[dict], concept_prompts: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    expected_ids = {seg["segment_id"] for seg in stage1_segments}
    missing = [seg["segment_id"] for seg in stage1_segments if seg["segment_id"] not in concept_prompts]
    if missing:
        raise ValueError(f"Missing concept prompts: {missing}")
    extra = [segment_id for segment_id in concept_prompts.keys() if segment_id not in expected_ids]
    ordered = {seg["segment_id"]: concept_prompts[seg["segment_id"]] for seg in stage1_segments}
    return ordered, extra


class PromptGenerationPipeline:
    """Application service for resolved context, concept prompts, and scene prompts."""

    required_keys = {"config", "app_config", "request", "stage1_segments"}
    produced_keys = {
        "global_context",
        "concept_prompts",
        "scene_details",
        "scene_prompts_json",
    }

    def __init__(
        self,
        *,
        llm_factory: LLMFactory,
        prompt_pipeline_factory: PromptPipelineFactory,
        concept_batcher_factory: ConceptBatcherFactory,
        scene_prompt_builder_factory: ScenePromptBuilderFactory,
    ):
        self.llm_factory = llm_factory
        self.prompt_pipeline_factory = prompt_pipeline_factory
        self.concept_batcher_factory = concept_batcher_factory
        self.scene_prompt_builder_factory = scene_prompt_builder_factory

    def execute(self, context: GenerateRenderPlanContext) -> GenerateRenderPlanContext:
        missing = self.required_keys - context.keys()
        if missing:
            raise KeyError(f"{self.__class__.__name__} missing context keys: {sorted(missing)}")
        return self.run(context)

    def run(self, context: GenerateRenderPlanContext) -> GenerateRenderPlanContext:
        config = context["config"]
        app_config = context["app_config"]
        request = context["request"]
        stage1_segments = context["stage1_segments"]
        resolved_context_json: Path = context["resolved_context_json"]
        concept_prompts_json: Path = context["concept_prompts_json"]
        scene_details_json: Path = context["scene_details_json"]
        scene_prompts_json: Path = context["scene_prompts_json"]
        log_step = context["log_step"]
        log_file = context["log_file"]
        run_spinner = context["run_spinner"]
        reporter = context["reporter"]
        artifact_store = context["artifact_store"]

        log_step("7. LLM Prompt Pipeline")
        llm = self.llm_factory(app_config)
        prompt_pipeline = self.prompt_pipeline_factory(llm)
        all_lyrics = " ".join(
            seg.get("lyrics", "")
            for seg in stage1_segments
            if seg.get("lyrics")
        ).strip()
        global_context = self.build_resolved_global_context(
            config=config,
            prompt_pipeline=prompt_pipeline,
            all_lyrics=all_lyrics,
            run_spinner=run_spinner,
            reporter=reporter,
        )
        prompt_pipeline.save_json(
            resolved_context_json,
            global_context,
            artifact_store=artifact_store,
        )
        log_file("Resolved Context JSON", resolved_context_json)
        reporter.panel(global_context["story_idea"], title="Story Idea")
        reporter.panel(global_context["style"], title="Style Block")
        reporter.table(
            "Resolved Subject / Locations",
            ["Field", "Value"],
            [
                ["Subject", global_context["subject"]],
                ["Locations", "\n".join(global_context["locations"])],
            ],
        )

        concept_story_input = join_notes(
            global_context["story_idea"],
            "STEERING:",
            get_steering_value(config, "concepts"),
        )
        if request.concept_batch_size > 0:
            reporter.message(
                f"[cyan]Using batched concept generation: "
                f"{request.concept_batch_size} segments per batch[/cyan]"
            )
            concept_batcher = self.concept_batcher_factory(llm, request.concept_batch_size)
            concept_prompts = run_spinner(
                f"Generating concept prompts in batches of {request.concept_batch_size}...",
                lambda: concept_batcher.create_concept_prompts_batched(
                    stage1_segments=stage1_segments,
                    story_idea=concept_story_input,
                    global_context=global_context,
                    notes=get_steering_value(config, "concepts"),
                ),
            )
        else:
            concept_prompts = run_spinner(
                "Generating concept prompts for all scenes...",
                lambda: call_with_supported_kwargs(
                    prompt_pipeline.create_concept_prompts,
                    stage1_segments=stage1_segments,
                    story_idea=concept_story_input,
                    global_context=global_context,
                    notes=get_steering_value(config, "concepts"),
                ),
            )

        concept_prompts, extra_concepts = validate_and_order_concept_prompts(stage1_segments, concept_prompts)
        if extra_concepts:
            reporter.message(f"[yellow]Ignoring extra concept prompt keys: {extra_concepts}[/yellow]")
        prompt_pipeline.save_json(
            concept_prompts_json,
            concept_prompts,
            artifact_store=artifact_store,
        )
        log_file("Concept Prompts JSON", concept_prompts_json)

        scene_details = run_spinner(
            "Generating camera and character motion per scene...",
            lambda: call_with_supported_kwargs(
                prompt_pipeline.create_scene_details,
                concept_prompts=concept_prompts,
                stage1_segments=stage1_segments,
                global_context=global_context,
            ),
        )
        prompt_pipeline.save_json(
            scene_details_json,
            scene_details,
            artifact_store=artifact_store,
        )
        log_file("Scene Details JSON", scene_details_json)

        log_step("8. Z-Image + LTX Scene Prompts")
        scene_prompt_builder = self.scene_prompt_builder_factory(llm)
        run_spinner(
            "Generating Z-Image and LTX prompts per scene...",
            lambda: scene_prompt_builder.build_scene_prompts(
                stage1_segments=stage1_segments,
                concept_prompts=concept_prompts,
                scene_details=scene_details,
                global_context=global_context,
                output_json_path=scene_prompts_json,
                zimage_instructions=get_steering_value(config, "zimage"),
                ltx_instructions=get_steering_value(config, "ltx"),
                trigger_word=str(get_config_value(config, "trigger_word", "") or ""),
                artifact_store=artifact_store,
            ),
        )
        log_file("Scene Prompts JSON", scene_prompts_json)

        context.update(
            {
                "global_context": global_context,
                "concept_prompts": concept_prompts,
                "scene_details": scene_details,
            }
        )
        return context

    def build_resolved_global_context(
        self,
        *,
        config: Any,
        prompt_pipeline: Any,
        all_lyrics: str,
        run_spinner: Callable[[str, Callable[[], Any]], Any],
        reporter: Any = None,
        console: Any = None,
    ) -> dict:
        if reporter is None and console is not None:
            reporter = console
        story_notes = join_notes(
            get_steering_value(config, "global_"),
            get_steering_value(config, "story_idea"),
        )
        style_notes = join_notes(
            get_steering_value(config, "global_"),
            get_steering_value(config, "style"),
        )
        subject_location_notes = join_notes(
            get_steering_value(config, "global_"),
            get_steering_value(config, "subject"),
            get_steering_value(config, "locations"),
        )

        config_story_idea = str(get_config_value(config, "story_idea", "") or "").strip()
        config_style = str(get_config_value(config, "style", "") or "").strip()
        config_subject = str(get_config_value(config, "subject", "") or "").strip()
        config_locations = get_config_value(config, "locations", []) or []
        config_actors = get_config_value(config, "actors", []) or []
        config_structured_locations = get_config_value(config, "structured_locations", []) or []
        subject_mode = str(get_config_value(config, "subject_mode", "multi") or "multi")
        max_scene_actors = int(get_config_value(config, "max_scene_actors", 1 if subject_mode == "single" else 4) or 4)
        silent_mode = bool(get_config_value(config, "silent_mode", False))

        story_idea = resolve_text_override(
            configured_value=config_story_idea,
            reporter=reporter,
            message="[yellow]Using story_idea override from project config.[/yellow]",
            generated_value_factory=lambda: run_spinner(
                "Generating story idea...",
                lambda: prompt_pipeline.create_story_idea(
                    lyrics=all_lyrics,
                    notes=story_notes,
                ),
            ),
        )

        style_block = resolve_text_override(
            configured_value=config_style,
            reporter=reporter,
            message="[yellow]Using style override from project config.[/yellow]",
            generated_value_factory=lambda: run_spinner(
                "Generating style block...",
                lambda: prompt_pipeline.create_style_block(
                    lyrics=all_lyrics,
                    notes=style_notes,
                ),
            ),
        )

        subject_locations = run_spinner(
            "Generating subject and locations fallback...",
            lambda: prompt_pipeline.create_subject_and_locations(
                story_idea=story_idea,
                notes=subject_location_notes,
            ),
        )

        subject = resolve_text_override(
            configured_value=config_subject,
            reporter=reporter,
            message="[yellow]Using subject override from project config.[/yellow]",
            generated_value_factory=lambda: subject_locations["subject"],
        )
        locations = resolve_locations_override(
            configured_locations=config_locations,
            generated_locations=subject_locations["locations"],
            reporter=reporter,
        )

        actors = config_items_as_dicts(config_actors) or config_items_as_dicts(subject_locations.get("actors", []))
        structured_locations = (
            config_items_as_dicts(config_structured_locations)
            or config_items_as_dicts(subject_locations.get("locations", []))
        )

        return {
            "story_idea": story_idea,
            "style": style_block,
            "subject": subject,
            "locations": locations,
            "actors": actors,
            "structured_locations": structured_locations,
            "subject_mode": subject_mode,
            "max_scene_actors": max_scene_actors,
            "silent_mode": silent_mode,
            "location_constraint": build_location_constraint(locations),
            "steering": {
                "global": get_steering_value(config, "global_"),
                "story_idea": get_steering_value(config, "story_idea"),
                "style": get_steering_value(config, "style"),
                "subject": get_steering_value(config, "subject"),
                "locations": get_steering_value(config, "locations"),
                "concepts": get_steering_value(config, "concepts"),
                "zimage": get_steering_value(config, "zimage"),
                "ltx": get_steering_value(config, "ltx"),
                "final_prompts": get_steering_value(config, "final_prompts"),
            },
            "prompt_guidance": config.prompt_guidance.as_prompt_context(),
        }
