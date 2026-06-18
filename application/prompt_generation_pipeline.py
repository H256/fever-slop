from __future__ import annotations

from pathlib import Path
import inspect
from typing import Any, Callable

from application.pipeline_context import GenerateRenderPlanContext
from rich.panel import Panel
from rich.table import Table

from adapters.openai_compatible_llm import OpenAICompatibleLLMClient
from concept_prompt_batcher import ConceptPromptBatcher
from prompt_pipeline import MusicVideoPromptPipeline
from scene_prompt_builder import ScenePromptBuilder


def join_notes(*parts: str) -> str:
    return "\n".join(part.strip() for part in parts if part and part.strip())


def get_steering_value(config: Any, name: str, default: str = "") -> str:
    steering = getattr(config, "steering", None)
    return str(getattr(steering, name, default) or "")


def get_config_value(config: Any, name: str, default: Any = None) -> Any:
    return getattr(config, name, default)


def call_with_supported_kwargs(func: Callable[..., Any], **kwargs):
    signature = inspect.signature(func)
    supported = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return func(**supported)


class PromptGenerationPipeline:
    """Application service for resolved context, concept prompts, and scene prompts."""

    required_keys = {"config", "app_config", "request", "stage1_segments"}
    produced_keys = {
        "global_context",
        "concept_prompts",
        "scene_details",
        "scene_prompts_json",
    }

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
        console = context["console"]

        log_step("7. LLM Prompt Pipeline")
        llm = OpenAICompatibleLLMClient(
            base_url=app_config.llm.base_url,
            model=app_config.llm.model,
            temperature=app_config.llm.temperature,
            max_tokens=app_config.llm.max_tokens,
        )
        prompt_pipeline = MusicVideoPromptPipeline(llm)
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
            console=console,
        )
        prompt_pipeline.save_json(resolved_context_json, global_context)
        log_file("Resolved Context JSON", resolved_context_json)
        console.print(Panel(global_context["story_idea"], title="Story Idea", border_style="green"))
        console.print(Panel(global_context["style"], title="Style Block", border_style="green"))

        context_table = Table(title="Resolved Subject / Locations")
        context_table.add_column("Field", style="bold")
        context_table.add_column("Value", style="cyan")
        context_table.add_row("Subject", global_context["subject"])
        context_table.add_row("Locations", "\n".join(global_context["locations"]))
        console.print(context_table)

        concept_story_input = join_notes(
            global_context["story_idea"],
            "STEERING:",
            get_steering_value(config, "concepts"),
        )
        if request.concept_batch_size > 0:
            console.print(
                f"[cyan]Using batched concept generation: "
                f"{request.concept_batch_size} segments per batch[/cyan]"
            )
            concept_batcher = ConceptPromptBatcher(
                llm=llm,
                batch_size=request.concept_batch_size,
            )
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

        expected_concept_ids = {seg["segment_id"] for seg in stage1_segments}
        missing_concepts = [
            seg["segment_id"]
            for seg in stage1_segments
            if seg["segment_id"] not in concept_prompts
        ]
        extra_concepts = [
            segment_id
            for segment_id in concept_prompts.keys()
            if segment_id not in expected_concept_ids
        ]
        if missing_concepts:
            raise ValueError(f"Missing concept prompts: {missing_concepts}")
        if extra_concepts:
            console.print(f"[yellow]Ignoring extra concept prompt keys: {extra_concepts}[/yellow]")
        concept_prompts = {
            seg["segment_id"]: concept_prompts[seg["segment_id"]]
            for seg in stage1_segments
        }
        prompt_pipeline.save_json(concept_prompts_json, concept_prompts)
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
        prompt_pipeline.save_json(scene_details_json, scene_details)
        log_file("Scene Details JSON", scene_details_json)

        log_step("8. Z-Image + LTX Scene Prompts")
        scene_prompt_builder = ScenePromptBuilder(llm)
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
        prompt_pipeline: MusicVideoPromptPipeline,
        all_lyrics: str,
        run_spinner: Callable[[str, Callable[[], Any]], Any],
        console: Any,
    ) -> dict:
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

        if config_story_idea:
            story_idea = config_story_idea
            console.print("[yellow]Using story_idea override from project config.[/yellow]")
        else:
            story_idea = run_spinner(
                "Generating story idea...",
                lambda: prompt_pipeline.create_story_idea(
                    lyrics=all_lyrics,
                    notes=story_notes,
                ),
            )

        if config_style:
            style_block = config_style
            console.print("[yellow]Using style override from project config.[/yellow]")
        else:
            style_block = run_spinner(
                "Generating style block...",
                lambda: prompt_pipeline.create_style_block(
                    lyrics=all_lyrics,
                    notes=style_notes,
                ),
            )

        subject_locations = run_spinner(
            "Generating subject and locations fallback...",
            lambda: prompt_pipeline.create_subject_and_locations(
                story_idea=story_idea,
                notes=subject_location_notes,
            ),
        )

        if config_subject:
            subject = config_subject
            console.print("[yellow]Using subject override from project config.[/yellow]")
        else:
            subject = subject_locations["subject"]

        if config_locations:
            locations = config_locations
            console.print("[yellow]Using locations override from project config.[/yellow]")
        else:
            locations = subject_locations["locations"]

        return {
            "story_idea": story_idea,
            "style": style_block,
            "subject": subject,
            "locations": locations,
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
