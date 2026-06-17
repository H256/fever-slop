from __future__ import annotations

from pathlib import Path
import argparse
import json

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from app_config import AppConfig
from comfyui_client import ComfyUIClient
from ltx_video_renderer import LTXVideoRenderer


console = Console()


def parse_scene_list(value: str | None) -> set[int] | None:
    if not value:
        return None
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            result.update(range(int(start_raw), int(end_raw) + 1))
        else:
            result.add(int(part))
    return result


def load_render_plan_subset(render_plan_path: str | Path, scene_numbers: set[int] | None, limit: int | None) -> list[dict]:
    render_plan = json.loads(Path(render_plan_path).read_text(encoding="utf-8"))
    if scene_numbers is not None:
        render_plan = [scene for scene in render_plan if int(scene["scene"]) in scene_numbers]
    if limit is not None:
        render_plan = render_plan[:limit]
    return render_plan


def main():
    parser = argparse.ArgumentParser(description="Render LTX I2V videos with rolling-frame preroll/tail trimming.")

    parser.add_argument("--app-config", default="./app_config.json")
    parser.add_argument("--render-plan", required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--storyboard-dir", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scenes", default=None)
    parser.add_argument("--no-skip-existing", action="store_true")

    parser.add_argument("--character-lora-strength", type=float, default=1.0)
    parser.add_argument("--randomize-seed", action="store_true")
    parser.add_argument("--seed-offset", type=int, default=100000)

    parser.add_argument("--no-upload-audio", action="store_true")
    parser.add_argument("--uploaded-audio-name", default=None)
    parser.add_argument("--no-upload-startframes", action="store_true")

    parser.add_argument("--segment-length-mode", choices=["frames_minus_one", "frames"], default="frames_minus_one")
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=10.0)
    parser.add_argument("--allow-out-of-range-clips", action="store_true")
    parser.add_argument("--debug-workflows-dir", default=None)

    parser.add_argument("--preroll-frames", type=int, default=6)
    parser.add_argument("--tail-loss-frames", type=int, default=6)
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--postprocess-streamcopy", action="store_true")

    args = parser.parse_args()

    app_config = AppConfig.load(args.app_config)
    scene_numbers = parse_scene_list(args.scenes)
    planned = load_render_plan_subset(args.render_plan, scene_numbers, args.limit)

    console.print(Panel.fit(
        f"[bold]LTX Video Renderer[/bold]\n\n"
        f"ComfyUI: [cyan]{app_config.comfyui.base_url}[/cyan]\n"
        f"Render plan: [cyan]{args.render_plan}[/cyan]\n"
        f"Workflow: [cyan]{args.workflow}[/cyan]\n"
        f"Audio: [cyan]{args.audio}[/cyan]\n"
        f"Storyboard: [cyan]{args.storyboard_dir}[/cyan]\n"
        f"Output: [cyan]{args.output_dir}[/cyan]\n"
        f"Scenes: [yellow]{len(planned)}[/yellow]\n"
        f"Preroll frames: [yellow]{args.preroll_frames}[/yellow]\n"
        f"Tail loss frames: [yellow]{args.tail_loss_frames}[/yellow]\n"
        f"Postprocess: [yellow]{not args.no_postprocess}[/yellow]",
        title="Startup",
        border_style="cyan",
    ))

    client = ComfyUIClient(base_url=app_config.comfyui.base_url)

    renderer = LTXVideoRenderer(
        client=client,
        ltx_workflow_path=args.workflow,
        output_dir=args.output_dir,
        character_lora_strength=args.character_lora_strength,
        randomize_seed=args.randomize_seed,
        seed_offset=args.seed_offset,
        segment_length_mode=args.segment_length_mode,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        allow_out_of_range_clips=args.allow_out_of_range_clips,
        debug_workflows_dir=args.debug_workflows_dir,
        preroll_frames=args.preroll_frames,
        tail_loss_frames=args.tail_loss_frames,
        postprocess=not args.no_postprocess,
        ffmpeg_path=args.ffmpeg,
        postprocess_reencode=not args.postprocess_streamcopy,
    )

    rendered: list[Path] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Rendering LTX scenes", total=len(planned))

        for scene in planned:
            scene_no = int(scene["scene"])
            progress.update(task, description=f"Rendering LTX scene {scene_no:04}")

            one_scene_plan = Path(args.output_dir) / "_single_scene_plan.json"
            one_scene_plan.parent.mkdir(parents=True, exist_ok=True)
            one_scene_plan.write_text(json.dumps([scene], ensure_ascii=False, indent=2), encoding="utf-8")

            files = renderer.render_videos(
                render_plan_path=one_scene_plan,
                audio_file=args.audio,
                storyboard_dir=args.storyboard_dir,
                skip_existing=not args.no_skip_existing,
                uploaded_audio_name=args.uploaded_audio_name,
                upload_audio=not args.no_upload_audio,
                upload_startframes=not args.no_upload_startframes,
            )

            rendered.extend(files)
            progress.advance(task)

    concat_file = Path(args.output_dir) / "concat_list.txt"
    console.print(f"[green]✓[/green] Rendered/available LTX clips: [yellow]{len(rendered)}[/yellow]")
    console.print(f"[green]✓[/green] FFmpeg concat list: [cyan]{concat_file}[/cyan]")
    console.print()
    console.print("Concat command:")
    console.print(f'[bold]ffmpeg -f concat -safe 0 -i "{concat_file}" -c copy "{Path(args.output_dir) / "final_concat.mp4"}"[/bold]')


if __name__ == "__main__":
    main()
