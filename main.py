from __future__ import annotations

from pathlib import Path
import argparse
import json
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from app_config import AppConfig
from project_config import ProjectConfig
from demucs_separator import DemucsSeparator
from prompt_relay_builder import build_scene_prompt_relay
from stage1_segment_builder import build_stage1_segment_json
from render_plan_builder import build_render_plan
from utils import save_timeline_json
from vocal_timeline_analyzer import (
    VocalTimelineAnalyzer,
    normalize_empty_vocals,
    merge_same_kind_segments,
)
from beat_analysis import (
    BeatImpactAnalyzer,
    BeatSceneDurationGenerator,
)
from llm_client import LocalOpenAIClient
from prompt_pipeline import MusicVideoPromptPipeline


console = Console()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def log_step(title: str):
    console.print()
    console.rule(f"[bold cyan]{title}[/bold cyan]")


def log_file(label: str, path: Path):
    console.print(f"[green]✓[/green] {label}: [cyan]{path}[/cyan]")


def run_spinner(description: str, func):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        progress.add_task(description, total=None)
        return func()


def join_notes(*parts: str) -> str:
    return "\n".join(part.strip() for part in parts if part and part.strip())


def build_resolved_global_context(
    *,
    config: ProjectConfig,
    prompt_pipeline: MusicVideoPromptPipeline,
    all_lyrics: str,
) -> dict:
    story_notes = join_notes(
        config.steering.global_,
        config.steering.story_idea,
    )

    style_notes = join_notes(
        config.steering.global_,
        config.steering.style,
    )

    subject_location_notes = join_notes(
        config.steering.global_,
        config.steering.subject,
        config.steering.locations,
    )

    if config.story_idea.strip():
        story_idea = config.story_idea.strip()
        console.print("[yellow]Using story_idea override from project config.[/yellow]")
    else:
        story_idea = run_spinner(
            "Generating story idea...",
            lambda: prompt_pipeline.create_story_idea(
                lyrics=all_lyrics,
                notes=story_notes,
            ),
        )

    if config.style.strip():
        style_block = config.style.strip()
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

    if config.subject.strip():
        subject = config.subject.strip()
        console.print("[yellow]Using subject override from project config.[/yellow]")
    else:
        subject = subject_locations["subject"]

    if config.locations:
        locations = config.locations
        console.print("[yellow]Using locations override from project config.[/yellow]")
    else:
        locations = subject_locations["locations"]

    return {
        "story_idea": story_idea,
        "style": style_block,
        "subject": subject,
        "locations": locations,
        "steering": {
            "global": config.steering.global_,
            "story_idea": config.steering.story_idea,
            "style": config.steering.style,
            "subject": config.steering.subject,
            "locations": config.steering.locations,
            "concepts": config.steering.concepts,
            "final_prompts": config.steering.final_prompts,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        required=True,
        help="Path to the project config.json.",
    )
    parser.add_argument(
        "--app-config",
        default="app_config.json",
        help="Path to global app_config.json. If missing, defaults are used.",
    )
    args = parser.parse_args()

    started_at = time.time()

    config = ProjectConfig.load(args.project)
    app_config = AppConfig.load(args.app_config)
    video_settings = config.to_video_settings()

    song_id = config.song_id

    stems_dir = config.stems_dir
    timeline_dir = config.timeline_dir
    prompts_dir = config.prompts_dir
    render_dir = config.render_dir

    timeline_json = timeline_dir / f"timeline_{song_id}.json"
    beat_json = timeline_dir / f"beat_data_{song_id}.json"
    scene_srt = timeline_dir / f"scenes_{song_id}.srt"
    stage1_segments_json = timeline_dir / f"stage1_segments_{song_id}.json"
    ltx_prompt_relay_json = prompts_dir / f"ltx_prompt_relay_{song_id}.json"

    resolved_context_json = prompts_dir / f"resolved_context_{song_id}.json"
    concept_prompts_json = prompts_dir / f"concept_prompts_{song_id}.json"
    scene_details_json = prompts_dir / f"scene_details_{song_id}.json"
    final_scene_prompts_json = prompts_dir / f"final_scene_prompts_{song_id}.json"
    render_plan_json = render_dir / f"render_plan_{song_id}.json"

    console.print(Panel.fit(
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

    # ------------------------------------------------------------------
    log_step("1. Demucs Stem Separation")

    separator = DemucsSeparator(model_name=config.audio.demucs_model)
    files = run_spinner(
        "Separating audio into vocals/drums/bass/other...",
        lambda: separator.separate(config.input_audio, stems_dir),
    )

    stem_table = Table(title="Generated Stems")
    stem_table.add_column("Stem", style="bold")
    stem_table.add_column("Path", style="cyan")
    for stem_name in ("vocals", "drums", "bass", "other"):
        stem_table.add_row(stem_name, str(files[stem_name]))
    console.print(stem_table)

    # ------------------------------------------------------------------
    log_step("2. Vocal Timeline Analysis")

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

    timeline = run_spinner(
        "Detecting vocal activity and transcribing lyrics...",
        lambda: analyzer.analyze(files["vocals"]),
    )
    timeline = normalize_empty_vocals(timeline)
    timeline = merge_same_kind_segments(timeline, merge_gap=vocal_cfg.merge_gap)

    save_timeline_json(timeline, timeline_json)
    log_file("Timeline JSON", timeline_json)

    vocal_count = sum(1 for seg in timeline if seg.kind == "vocals")
    instrumental_count = sum(1 for seg in timeline if seg.kind == "instrumental")
    console.print(
        f"[green]✓[/green] Timeline segments: "
        f"[yellow]{len(timeline)}[/yellow] total, "
        f"[yellow]{vocal_count}[/yellow] vocals, "
        f"[yellow]{instrumental_count}[/yellow] instrumental"
    )

    # ------------------------------------------------------------------
    log_step("3. Beat / Impact Analysis")

    beat_analyzer = BeatImpactAnalyzer()
    run_spinner(
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

    log_file("Beat Data JSON", beat_json)
    beat_data = read_json(beat_json)
    console.print(
        f"[green]✓[/green] BPM: [yellow]{beat_data.get('bpm')}[/yellow], "
        f"beats: [yellow]{len(beat_data.get('beats', []))}[/yellow], "
        f"source: [yellow]{beat_data.get('source_used_for_beats')}[/yellow]"
    )

    # ------------------------------------------------------------------
    log_step("4. Beat-Aligned Scene SRT")

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
        output_srt_path=scene_srt,
    )
    log_file("Scene SRT", scene_srt)

    # ------------------------------------------------------------------
    log_step("5. Stage 1 Segment Mapping")

    build_stage1_segment_json(
        scene_srt_file=scene_srt,
        vocal_timeline_json=timeline_json,
        output_json_file=stage1_segments_json,
    )
    log_file("Stage 1 Segments JSON", stage1_segments_json)

    stage1_segments = read_json(stage1_segments_json)
    type_counts = {}
    for seg in stage1_segments:
        type_counts[seg["type"]] = type_counts.get(seg["type"], 0) + 1

    console.print(
        f"[green]✓[/green] Stage 1 segments: [yellow]{len(stage1_segments)}[/yellow] "
        f"{type_counts}"
    )

    # ------------------------------------------------------------------
    log_step("6. LTX Prompt Relay Skeleton")

    build_scene_prompt_relay(
        scene_srt_file=scene_srt,
        vocal_timeline_json=timeline_json,
        output_json_file=ltx_prompt_relay_json,
        video_settings=video_settings,
    )
    log_file("LTX Prompt Relay JSON", ltx_prompt_relay_json)

    # ------------------------------------------------------------------
    log_step("7. LLM Prompt Pipeline")

    llm = LocalOpenAIClient(
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

    global_context = build_resolved_global_context(
        config=config,
        prompt_pipeline=prompt_pipeline,
        all_lyrics=all_lyrics,
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
        config.steering.concepts,
    )

    concept_prompts = run_spinner(
        "Generating concept prompts for all scenes...",
        lambda: prompt_pipeline.create_concept_prompts(
            stage1_segments=stage1_segments,
            story_idea=concept_story_input,
        ),
    )

    prompt_pipeline.save_json(concept_prompts_json, concept_prompts)
    log_file("Concept Prompts JSON", concept_prompts_json)

    scene_details = run_spinner(
        "Generating camera and character motion per scene...",
        lambda: prompt_pipeline.create_scene_details(
            concept_prompts=concept_prompts,
        ),
    )

    prompt_pipeline.save_json(scene_details_json, scene_details)
    log_file("Scene Details JSON", scene_details_json)

    if config.steering.final_prompts:
        global_context["final_prompt_steering"] = config.steering.final_prompts

    final_scene_prompts = run_spinner(
        "Generating final scene prompts...",
        lambda: prompt_pipeline.create_final_scene_prompts(
            stage1_segments=stage1_segments,
            concept_prompts=concept_prompts,
            scene_details=scene_details,
            global_context=global_context,
        ),
    )

    prompt_pipeline.save_json(final_scene_prompts_json, final_scene_prompts)
    log_file("Final Scene Prompts JSON", final_scene_prompts_json)

    # ------------------------------------------------------------------
    log_step("8. Render Plan")

    build_render_plan(
        final_scene_prompts_json=final_scene_prompts_json,
        ltx_prompt_relay_json=ltx_prompt_relay_json,
        output_json_file=render_plan_json,
        video_settings=video_settings,
    )

    log_file("Render Plan JSON", render_plan_json)

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
    console.print(summary)

    elapsed = time.time() - started_at
    console.print(Panel.fit(
        f"[bold green]Done.[/bold green]\n\n"
        f"Elapsed: [yellow]{elapsed:.1f}s[/yellow]\n"
        f"Render plan: [cyan]{render_plan_json}[/cyan]",
        title="Pipeline Complete",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
