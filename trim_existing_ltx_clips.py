from __future__ import annotations

from pathlib import Path
import argparse
import json

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from video_postprocessor import VideoPostProcessor, TrimSpec


console = Console()


def parse_scene_list(value: str | None) -> set[int] | None:
    if not value:
        return None
    result = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            result.update(range(int(a), int(b) + 1))
        else:
            result.add(int(part))
    return result


def main():
    parser = argparse.ArgumentParser(description="Trim existing raw LTX clips with rolling-frame parameters.")
    parser.add_argument("--render-plan", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preroll-frames", type=int, default=6)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--streamcopy", action="store_true")
    parser.add_argument("--scenes", default=None)
    args = parser.parse_args()

    plan = json.loads(Path(args.render_plan).read_text(encoding="utf-8"))
    scene_numbers = parse_scene_list(args.scenes)
    if scene_numbers:
        plan = [s for s in plan if int(s["scene"]) in scene_numbers]

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = VideoPostProcessor(ffmpeg_path=args.ffmpeg, reencode=not args.streamcopy)

    outputs = []
    manifest = []

    console.print(Panel.fit(
        f"[bold]Trim Existing LTX Clips[/bold]\n\n"
        f"Raw: [cyan]{raw_dir}[/cyan]\n"
        f"Output: [cyan]{output_dir}[/cyan]\n"
        f"Scenes: [yellow]{len(plan)}[/yellow]\n"
        f"Preroll: [yellow]{args.preroll_frames}[/yellow]",
        title="Startup",
        border_style="cyan",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Trimming clips", total=len(plan))

        for scene in plan:
            scene_no = int(scene["scene"])
            raw_file = raw_dir / f"scene_{scene_no:04}_raw.mp4"
            if not raw_file.exists():
                raw_file = raw_dir / f"scene_{scene_no:04}.mp4"
            if not raw_file.exists():
                raise FileNotFoundError(raw_file)

            preroll = 0 if scene_no == 1 else args.preroll_frames
            output_file = output_dir / f"scene_{scene_no:04}.mp4"

            spec = TrimSpec(
                source_file=raw_file,
                output_file=output_file,
                fps=int(scene["fps"]),
                trim_front_frames=preroll,
                keep_frames=int(scene["frame_count"]),
                scene=scene_no,
            )

            progress.update(task, description=f"Trimming scene {scene_no:04}")
            processor.trim_clip(spec)
            outputs.append(output_file)
            manifest.append({"scene": scene_no, "raw": str(raw_file), "output": str(output_file), "trim_front_frames": preroll})
            progress.advance(task)

    concat = processor.write_concat_list(outputs, output_dir / "concat_list.txt")
    processor.write_manifest(manifest, output_dir / "trim_manifest.json")
    console.print(f"[green]✓[/green] Trimmed clips: [yellow]{len(outputs)}[/yellow]")
    console.print(f"[green]✓[/green] Concat list: [cyan]{concat}[/cyan]")


if __name__ == "__main__":
    main()
