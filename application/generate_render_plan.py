from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import inspect
import json
import time
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from application.audio_timeline_pipeline import AudioTimelinePipeline
from application.prompt_generation_pipeline import PromptGenerationPipeline
from application.render_plan_pipeline import RenderPlanPipeline
from application.scene_timeline_pipeline import SceneTimelinePipeline
from adapters.openai_compatible_llm import OpenAICompatibleLLMClient
from app_config import AppConfig
from beat_analysis import BeatImpactAnalyzer, BeatSceneDurationGenerator
from concept_prompt_batcher import ConceptPromptBatcher
from demucs_separator import DemucsSeparator
from project_config import ProjectConfig, ProjectPaths
from prompt_pipeline import MusicVideoPromptPipeline
from prompt_relay_builder import build_scene_prompt_relay
from render_plan_builder import build_render_plan
from scene_duration_enforcer import (
    enforce_scene_srt_file,
    parse_scene_srt,
    validate_scene_durations,
)
from scene_prompt_builder import ScenePromptBuilder
from stage1_segment_builder import build_stage1_segment_json
from utils import save_timeline_json
from vocal_timeline_analyzer import (
    VocalTimelineAnalyzer,
    merge_same_kind_segments,
    normalize_empty_vocals,
)


@dataclass(frozen=True)
class GenerateRenderPlanRequest:
    project_config_path: Path
    app_config_path: Path = Path("app_config.json")
    concept_batch_size: int = 0
    render_storyboard: bool = False
    zimage_workflow_path: Path | None = None


@dataclass(frozen=True)
class GenerateRenderPlanResult:
    render_plan_path: Path
    scene_count: int
    total_frames: int
    total_duration_seconds: float


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def join_notes(*parts: str) -> str:
    return "\n".join(part.strip() for part in parts if part and part.strip())


def get_steering_value(config: ProjectConfig, name: str, default: str = "") -> str:
    steering = getattr(config, "steering", None)
    return str(getattr(steering, name, default) or "")


def get_config_value(config: ProjectConfig, name: str, default: Any = None) -> Any:
    return getattr(config, name, default)


def call_with_supported_kwargs(func: Callable[..., Any], **kwargs):
    signature = inspect.signature(func)
    supported = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return func(**supported)


class GenerateRenderPlanUseCase:
    def __init__(self, console: Console | None = None, pipeline_services: list[Any] | None = None):
        self.console = console or Console()
        self.pipeline_services = pipeline_services or [
            AudioTimelinePipeline(),
            SceneTimelinePipeline(),
            PromptGenerationPipeline(),
            RenderPlanPipeline(),
        ]

    def execute_services(self, context: dict[str, Any]) -> dict[str, Any]:
        for service in self.pipeline_services:
            context = service.execute(context)
        return context

    def log_step(self, title: str):
        self.console.print()
        self.console.rule(f"[bold cyan]{title}[/bold cyan]")

    def log_file(self, label: str, path: Path):
        self.console.print(f"[green]✓[/green] {label}: [cyan]{path}[/cyan]")

    def run_spinner(self, description: str, func: Callable[[], Any]):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            progress.add_task(description, total=None)
            return func()

    def build_resolved_global_context(
        self,
        *,
        config: ProjectConfig,
        prompt_pipeline: MusicVideoPromptPipeline,
        all_lyrics: str,
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
            self.console.print("[yellow]Using story_idea override from project config.[/yellow]")
        else:
            story_idea = self.run_spinner(
                "Generating story idea...",
                lambda: prompt_pipeline.create_story_idea(
                    lyrics=all_lyrics,
                    notes=story_notes,
                ),
            )

        if config_style:
            style_block = config_style
            self.console.print("[yellow]Using style override from project config.[/yellow]")
        else:
            style_block = self.run_spinner(
                "Generating style block...",
                lambda: prompt_pipeline.create_style_block(
                    lyrics=all_lyrics,
                    notes=style_notes,
                ),
            )

        subject_locations = self.run_spinner(
            "Generating subject and locations fallback...",
            lambda: prompt_pipeline.create_subject_and_locations(
                story_idea=story_idea,
                notes=subject_location_notes,
            ),
        )

        if config_subject:
            subject = config_subject
            self.console.print("[yellow]Using subject override from project config.[/yellow]")
        else:
            subject = subject_locations["subject"]

        if config_locations:
            locations = config_locations
            self.console.print("[yellow]Using locations override from project config.[/yellow]")
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

    def execute(self, request: GenerateRenderPlanRequest) -> GenerateRenderPlanResult:
        started_at = time.time()

        config = ProjectConfig.load(request.project_config_path)
        paths = ProjectPaths.from_config(config)
        app_config = AppConfig.load(request.app_config_path)
        video_settings = config.to_video_settings()

        song_id = getattr(config, "song_id", None) or getattr(config, "project_name", "") or config.input_audio.stem

        stems_dir = paths.stems_dir
        timeline_dir = paths.timeline_dir
        prompts_dir = paths.prompts_dir
        render_dir = paths.render_dir
        paths.ensure_output_dirs()

        timeline_json = timeline_dir / f"timeline_{song_id}.json"
        beat_json = timeline_dir / f"beat_data_{song_id}.json"
        scene_srt_raw = timeline_dir / f"scenes_{song_id}_raw.srt"
        scene_srt = timeline_dir / f"scenes_{song_id}.srt"
        stage1_segments_json = timeline_dir / f"stage1_segments_{song_id}.json"
        ltx_prompt_relay_json = prompts_dir / f"ltx_prompt_relay_{song_id}.json"

        resolved_context_json = prompts_dir / f"resolved_context_{song_id}.json"
        concept_prompts_json = prompts_dir / f"concept_prompts_{song_id}.json"
        scene_details_json = prompts_dir / f"scene_details_{song_id}.json"
        scene_prompts_json = prompts_dir / f"scene_prompts_{song_id}.json"
        render_plan_json = render_dir / f"render_plan_{song_id}.json"

        self.console.print(Panel.fit(
            f"[bold]Music Video Pipeline[/bold]\n\n"
            f"Project: [cyan]{config.project_name}[/cyan]\n"
            f"Input: [cyan]{config.input_audio}[/cyan]\n"
            f"Output: [cyan]{config.output_dir}[/cyan]\n"
            f"FPS: [yellow]{video_settings.fps}[/yellow]\n"
            f"Resolution: [yellow]{video_settings.width}x{video_settings.height}[/yellow]\n"
            f"LLM: [yellow]{app_config.llm.model}[/yellow] @ [cyan]{app_config.llm.base_url}[/cyan]",
            title="Startup",
            border_style="cyan",
        ))

        if not config.input_audio.exists():
            raise FileNotFoundError(config.input_audio)

        self.log_step("1. Demucs Stem Separation")
        separator = DemucsSeparator(model_name=config.audio.demucs_model)
        files = self.run_spinner(
            "Separating audio into vocals/drums/bass/other...",
            lambda: separator.separate(config.input_audio, stems_dir),
        )

        stem_table = Table(title="Generated Stems")
        stem_table.add_column("Stem", style="bold")
        stem_table.add_column("Path", style="cyan")
        for stem_name in ("vocals", "drums", "bass", "other"):
            stem_table.add_row(stem_name, str(files[stem_name]))
        self.console.print(stem_table)

        self.log_step("2. Vocal Timeline Analysis")
        vocal_cfg = config.vocal_detection
        analyzer = VocalTimelineAnalyzer(
            whisper_model=config.audio.whisper_model,
            language=config.audio.language,
            merge_gap=vocal_cfg.merge_gap,
            min_vocal_duration=vocal_cfg.min_vocal_duration,
            min_silence_duration=vocal_cfg.min_silence_duration,
            rms_low_percentile=vocal_cfg.rms_low_percentile,
            rms_high_percentile=vocal_cfg.rms_high_percentile,
            rms_ratio=vocal_cfg.rms_ratio,
            smooth_frames=vocal_cfg.smooth_frames,
        )
        timeline = self.run_spinner(
            "Detecting vocal activity and transcribing lyrics...",
            lambda: analyzer.analyze(files["vocals"]),
        )
        timeline = normalize_empty_vocals(timeline)
        timeline = merge_same_kind_segments(timeline, merge_gap=vocal_cfg.merge_gap)
        save_timeline_json(timeline, timeline_json)
        self.log_file("Timeline JSON", timeline_json)

        vocal_count = sum(1 for seg in timeline if seg.kind == "vocals")
        instrumental_count = sum(1 for seg in timeline if seg.kind == "instrumental")
        self.console.print(
            f"[green]✓[/green] Timeline segments: "
            f"[yellow]{len(timeline)}[/yellow] total, "
            f"[yellow]{vocal_count}[/yellow] vocals, "
            f"[yellow]{instrumental_count}[/yellow] instrumental"
        )

        self.log_step("3. Beat / Impact Analysis")
        beat_analyzer = BeatImpactAnalyzer()
        self.run_spinner(
            "Analyzing beats and impact values...",
            lambda: beat_analyzer.analyze_to_json_file(
                final_mix_path=config.input_audio,
                output_json_path=beat_json,
                drums_path=files["drums"],
                bass_path=files["bass"],
                vocals_path=files["vocals"],
                other_path=files["other"],
            ),
        )
        self.log_file("Beat Data JSON", beat_json)
        beat_data = read_json(beat_json)
        self.console.print(
            f"[green]✓[/green] BPM: [yellow]{beat_data.get('bpm')}[/yellow], "
            f"beats: [yellow]{len(beat_data.get('beats', []))}[/yellow], "
            f"source: [yellow]{beat_data.get('source_used_for_beats')}[/yellow]"
        )

        self.log_step("4. Beat-Aligned Scene SRT")
        scene_cfg = config.scene_generation
        scene_generator = BeatSceneDurationGenerator(
            min_duration=scene_cfg.min_duration,
            max_duration=scene_cfg.max_duration,
            bias=scene_cfg.bias,
            duration_preset=scene_cfg.duration_preset,
            seed=scene_cfg.seed,
        )
        scene_generator.generate_from_json_file(
            beat_json_path=beat_json,
            output_srt_path=scene_srt_raw,
        )
        self.log_file("Raw Scene SRT", scene_srt_raw)
        enforce_scene_srt_file(
            input_srt=scene_srt_raw,
            output_srt=scene_srt,
            min_duration=scene_cfg.min_duration,
            max_duration=scene_cfg.max_duration,
        )
        self.log_file("Repaired Scene SRT", scene_srt)
        repaired_scenes = parse_scene_srt(scene_srt)
        duration_errors = validate_scene_durations(
            repaired_scenes,
            min_duration=scene_cfg.min_duration,
            max_duration=scene_cfg.max_duration,
        )
        if duration_errors:
            raise ValueError(
                "Scene duration constraints failed after repair:\n"
                + "\n".join(duration_errors)
            )
        shortest_scene = min((scene.duration for scene in repaired_scenes), default=0.0)
        longest_scene = max((scene.duration for scene in repaired_scenes), default=0.0)
        self.console.print(
            f"[green]✓[/green] Scene duration range: "
            f"[yellow]{shortest_scene:.2f}s[/yellow].."
            f"[yellow]{longest_scene:.2f}s[/yellow] "
            f"from [yellow]{len(repaired_scenes)}[/yellow] scenes"
        )

        self.log_step("5. Stage 1 Segment Mapping")
        build_stage1_segment_json(
            scene_srt_file=scene_srt,
            vocal_timeline_json=timeline_json,
            output_json_file=stage1_segments_json,
        )
        self.log_file("Stage 1 Segments JSON", stage1_segments_json)
        stage1_segments = read_json(stage1_segments_json)
        type_counts: dict[str, int] = {}
        for seg in stage1_segments:
            type_counts[seg["type"]] = type_counts.get(seg["type"], 0) + 1
        self.console.print(
            f"[green]✓[/green] Stage 1 segments: [yellow]{len(stage1_segments)}[/yellow] "
            f"{type_counts}"
        )

        self.log_step("6. LTX Prompt Relay Skeleton")
        build_scene_prompt_relay(
            scene_srt_file=scene_srt,
            vocal_timeline_json=timeline_json,
            output_json_file=ltx_prompt_relay_json,
            video_settings=video_settings,
        )
        self.log_file("LTX Prompt Relay JSON", ltx_prompt_relay_json)

        self.log_step("7. LLM Prompt Pipeline")
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
        )
        prompt_pipeline.save_json(resolved_context_json, global_context)
        self.log_file("Resolved Context JSON", resolved_context_json)
        self.console.print(Panel(global_context["story_idea"], title="Story Idea", border_style="green"))
        self.console.print(Panel(global_context["style"], title="Style Block", border_style="green"))

        context_table = Table(title="Resolved Subject / Locations")
        context_table.add_column("Field", style="bold")
        context_table.add_column("Value", style="cyan")
        context_table.add_row("Subject", global_context["subject"])
        context_table.add_row("Locations", "\n".join(global_context["locations"]))
        self.console.print(context_table)

        concept_story_input = join_notes(
            global_context["story_idea"],
            "STEERING:",
            get_steering_value(config, "concepts"),
        )
        if request.concept_batch_size > 0:
            self.console.print(
                f"[cyan]Using batched concept generation: "
                f"{request.concept_batch_size} segments per batch[/cyan]"
            )
            concept_batcher = ConceptPromptBatcher(
                llm=llm,
                batch_size=request.concept_batch_size,
            )
            concept_prompts = self.run_spinner(
                f"Generating concept prompts in batches of {request.concept_batch_size}...",
                lambda: concept_batcher.create_concept_prompts_batched(
                    stage1_segments=stage1_segments,
                    story_idea=concept_story_input,
                    global_context=global_context,
                    notes=get_steering_value(config, "concepts"),
                ),
            )
        else:
            concept_prompts = self.run_spinner(
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
            self.console.print(f"[yellow]Ignoring extra concept prompt keys: {extra_concepts}[/yellow]")
        concept_prompts = {
            seg["segment_id"]: concept_prompts[seg["segment_id"]]
            for seg in stage1_segments
        }
        prompt_pipeline.save_json(concept_prompts_json, concept_prompts)
        self.log_file("Concept Prompts JSON", concept_prompts_json)

        scene_details = self.run_spinner(
            "Generating camera and character motion per scene...",
            lambda: call_with_supported_kwargs(
                prompt_pipeline.create_scene_details,
                concept_prompts=concept_prompts,
                stage1_segments=stage1_segments,
                global_context=global_context,
            ),
        )
        prompt_pipeline.save_json(scene_details_json, scene_details)
        self.log_file("Scene Details JSON", scene_details_json)

        self.log_step("8. Z-Image + LTX Scene Prompts")
        scene_prompt_builder = ScenePromptBuilder(llm)
        self.run_spinner(
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
        self.log_file("Scene Prompts JSON", scene_prompts_json)

        self.log_step("9. Render Plan")
        build_render_plan(
            scene_prompts_json=scene_prompts_json,
            ltx_prompt_relay_json=ltx_prompt_relay_json,
            output_json_file=render_plan_json,
            video_settings=video_settings,
        )
        self.log_file("Render Plan JSON", render_plan_json)

        render_plan = read_json(render_plan_json)
        total_frames = sum(scene["frame_count"] for scene in render_plan)
        total_duration = sum(scene["duration_seconds"] for scene in render_plan)
        summary = Table(title="Render Plan Summary")
        summary.add_column("Metric", style="bold")
        summary.add_column("Value", style="yellow")
        summary.add_row("Scenes / Cuts", str(len(render_plan)))
        summary.add_row("Total Frames", str(total_frames))
        summary.add_row("Total Duration", f"{total_duration:.2f}s")
        summary.add_row("FPS", str(video_settings.fps))
        summary.add_row("Resolution", f"{video_settings.width}x{video_settings.height}")
        self.console.print(summary)

        elapsed = time.time() - started_at
        self.console.print(Panel.fit(
            f"[bold green]Done.[/bold green]\n\n"
            f"Elapsed: [yellow]{elapsed:.1f}s[/yellow]\n"
            f"Render plan: [cyan]{render_plan_json}[/cyan]",
            title="Pipeline Complete",
            border_style="green",
        ))

        if request.render_storyboard:
            if not request.zimage_workflow_path:
                raise ValueError(
                    "--zimage-workflow is required when --render-storyboard is used"
                )

            from comfyui_client import ComfyUIClient
            from storyboard_renderer import StoryboardRenderer

            client = ComfyUIClient(base_url=app_config.comfyui.base_url)
            renderer = StoryboardRenderer(
                client=client,
                zimage_workflow_path=request.zimage_workflow_path,
                output_dir=render_dir / "storyboard",
                positive_prompt_node_title="#POSITIVE_PROMPT",
                negative_prompt_node_title="#NEGATIVE_PROMPT",
                save_image_node_title="#SAVE_IMAGE",
                character_lora_node_title="#CHARACTER_LORA",
            )
            rendered = renderer.render_storyboard(render_plan_path=render_plan_json)
            self.console.print(
                f"[green]✓[/green] Rendered storyboard frames: [yellow]{len(rendered)}[/yellow]"
            )

        return GenerateRenderPlanResult(
            render_plan_path=render_plan_json,
            scene_count=len(render_plan),
            total_frames=total_frames,
            total_duration_seconds=total_duration,
        )
