from __future__ import annotations

from pathlib import Path
import argparse
import json

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from feverslop.adapters.comfyui_video_backend import ComfyUIVideoRenderBackend
from feverslop.adapters.comfyui_model_resolver import ComfyUIModelResolver
from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.application.render_video import RenderVideoScenesRequest, RenderVideoScenesUseCase
from feverslop.config.app_config import AppConfig
from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.config.project_config import ProjectConfig
from feverslop.ports.rendering import WorkflowAnchorConfig


console = Console()


ROLLING_FRAME_PROFILES = {
    "original": (50, 25, True),
    "safe": (6, 0, False),
    "off": (0, 0, False),
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render LTX I2V videos with optional rolling-frame preroll/tail trimming.")

    parser.add_argument("--app-config", default="./app_config.json")
    parser.add_argument("--project-config", default=None)
    parser.add_argument("--render-plan", required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument(
        "--render-mode",
        choices=["relay", "single_prompt", "auto"],
        default="single_prompt",
        help="single_prompt uses #PROMPT, relay uses #PROMPT_RELAY, auto uses ltx.render_mode_hint per scene.",
    )
    parser.add_argument(
        "--single-prompt-workflow",
        default=None,
        help="Workflow for single_prompt/auto scenes. If omitted, --workflow is used.",
    )
    parser.add_argument("--single-prompt-title", default="#PROMPT")
    parser.add_argument("--single-prompt-input", default="text")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--storyboard-dir", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scenes", default=None)
    parser.add_argument("--no-skip-existing", action="store_true")

    parser.add_argument("--character-lora-strength", type=float, default=1.0)
    parser.add_argument("--lora-1-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--lora-1-name", default=None)
    parser.add_argument("--lora-1-strength-model", type=float, default=None)
    parser.add_argument("--lora-1-strength-clip", type=float, default=None)
    parser.add_argument("--randomize-seed", action="store_true")
    parser.add_argument("--seed-offset", type=int, default=100000)

    parser.add_argument("--no-upload-audio", action="store_true")
    parser.add_argument("--uploaded-audio-name", default=None)
    parser.add_argument("--no-upload-startframes", action="store_true")

    parser.add_argument("--segment-length-mode", choices=["frames_minus_one", "frames"], default="frames_minus_one")
    parser.add_argument("--min-duration", type=float, default=None)
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument("--allow-out-of-range-clips", action="store_true")
    parser.add_argument("--debug-workflows-dir", default=None)

    parser.add_argument(
        "--rolling-frame-profile",
        choices=sorted(ROLLING_FRAME_PROFILES),
        default="original",
        help="original=pre 50/tail 25 plus 8N+1 rounding, safe=6/0 no rounding, off=0/0.",
    )
    parser.add_argument("--preroll-frames", type=int, default=None)
    parser.add_argument("--tail-loss-frames", type=int, default=None)
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--postprocess-streamcopy", action="store_true")
    return parser


def _discover_project_config_path(render_plan_path: str | Path) -> Path | None:
    render_plan_path = Path(render_plan_path).resolve()
    for parent in render_plan_path.parents:
        candidate = parent / "config.json"
        if candidate.exists():
            return candidate
    return None


def resolve_project_config_defaults(args: argparse.Namespace) -> dict:
    project_config_path = args.project_config
    if not project_config_path and getattr(args, "render_plan", None):
        discovered = _discover_project_config_path(args.render_plan)
        if discovered:
            project_config_path = str(discovered)

    project_config = ProjectConfig.load(project_config_path) if project_config_path else None
    scene_generation = project_config.scene_generation if project_config else None
    lora_1 = project_config.lora_1 if project_config else None

    min_duration = args.min_duration if args.min_duration is not None else (scene_generation.min_duration if scene_generation else 2.0)
    max_duration = args.max_duration if args.max_duration is not None else (scene_generation.max_duration if scene_generation else 10.0)

    lora_1_enabled = args.lora_1_enabled if args.lora_1_enabled is not None else (lora_1.enabled if lora_1 else False)
    lora_1_name = args.lora_1_name if args.lora_1_name is not None else (lora_1.name if lora_1 else "")
    lora_1_strength_model = (
        args.lora_1_strength_model
        if args.lora_1_strength_model is not None
        else (lora_1.strength_model if lora_1 else 1.0)
    )
    lora_1_strength_clip = (
        args.lora_1_strength_clip
        if args.lora_1_strength_clip is not None
        else (lora_1.strength_clip if lora_1 else 1.0)
    )
    lora_1_strengths_explicit = (
        args.lora_1_strength_model is not None
        or args.lora_1_strength_clip is not None
    )

    return {
        "project_config": project_config,
        "min_duration": float(min_duration),
        "max_duration": float(max_duration),
        "lora_1_enabled": bool(lora_1_enabled),
        "lora_1_name": str(lora_1_name or ""),
        "lora_1_strength_model": float(lora_1_strength_model),
        "lora_1_strength_clip": float(lora_1_strength_clip),
        "lora_1_strengths_explicit": bool(lora_1_strengths_explicit),
    }


def resolve_rolling_frames(args: argparse.Namespace) -> tuple[int, int, bool]:
    profile_preroll, profile_tail, profile_rounding = ROLLING_FRAME_PROFILES[args.rolling_frame_profile]
    preroll = profile_preroll if args.preroll_frames is None else args.preroll_frames
    tail = profile_tail if args.tail_loss_frames is None else args.tail_loss_frames
    return max(0, int(preroll)), max(0, int(tail)), bool(profile_rounding)


def rewrite_concat_list(rendered_files: list[Path], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    concat_file = output_dir / "concat_list.txt"
    with concat_file.open("w", encoding="utf-8") as f:
        for path in rendered_files:
            f.write(f"file '{Path(path).resolve().as_posix()}'\n")
    return concat_file


def sanitize_file_stem(value: str | None, fallback: str = "final_concat") -> str:
    raw = str(value or "").strip() or fallback
    safe = "".join(char if char.isascii() and (char.isalnum() or char in "._-") else "_" for char in raw)
    safe = safe.strip("._-")
    return safe or fallback


def final_concat_paths(output_dir: str | Path, project_name: str | None = None) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    if project_name:
        file_stem = sanitize_file_stem(project_name, "final_concat")
        return output_dir / f"{file_stem}_video_only.mp4", output_dir / f"{file_stem}.mp4"

    return output_dir / "final_concat_video_only.mp4", output_dir / "final_concat.mp4"


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
    parser = build_arg_parser()
    args = parser.parse_args()
    resolved = resolve_project_config_defaults(args)

    if args.render_mode == "auto" and not args.single_prompt_workflow:
        raise ValueError("--single-prompt-workflow is required when --render-mode auto is used")
    if resolved["lora_1_enabled"] and not resolved["lora_1_name"]:
        raise ValueError("--lora-1-name is required when --lora-1-enabled is used")
    preroll_frames, tail_loss_frames, round_render_frames_to_8n1 = resolve_rolling_frames(args)

    app_config = AppConfig.load(args.app_config)
    scene_numbers = parse_scene_list(args.scenes)
    planned = load_render_plan_subset(args.render_plan, scene_numbers, args.limit)

    console.print(Panel.fit(
        f"[bold]LTX Video Renderer[/bold]\n\n"
        f"ComfyUI: [cyan]{app_config.comfyui.base_url}[/cyan]\n"
        f"Render plan: [cyan]{args.render_plan}[/cyan]\n"
        f"Workflow: [cyan]{args.workflow}[/cyan]\n"
        f"Render mode: [yellow]{args.render_mode}[/yellow]\n"
        f"Single-prompt workflow: [cyan]{args.single_prompt_workflow or args.workflow}[/cyan]\n"
        f"Audio: [cyan]{args.audio}[/cyan]\n"
        f"Storyboard: [cyan]{args.storyboard_dir}[/cyan]\n"
        f"Output: [cyan]{args.output_dir}[/cyan]\n"
        f"Scenes: [yellow]{len(planned)}[/yellow]\n"
        f"Rolling profile: [yellow]{args.rolling_frame_profile}[/yellow]\n"
        f"Preroll frames: [yellow]{preroll_frames}[/yellow]\n"
        f"Tail loss frames: [yellow]{tail_loss_frames}[/yellow]\n"
        f"Round render frames to 8N+1: [yellow]{round_render_frames_to_8n1}[/yellow]\n"
        f"LoRA 1 enabled: [yellow]{resolved['lora_1_enabled']}[/yellow]\n"
        f"LoRA 1 name: [cyan]{resolved['lora_1_name']}[/cyan]\n"
        f"Postprocess: [yellow]{not args.no_postprocess}[/yellow]",
        title="Startup",
        border_style="cyan",
    ))

    client = ComfyUIClient(base_url=app_config.comfyui.base_url)
    model_resolver = ComfyUIModelResolver(
        client,
        overrides=app_config.comfyui.model_overrides,
    )

    backend = ComfyUIVideoRenderBackend(
        client=client,
        ltx_workflow_path=args.workflow,
        output_dir=args.output_dir,
        single_prompt_workflow_path=args.single_prompt_workflow,
        render_mode=args.render_mode,
        single_prompt_node_title=args.single_prompt_title,
        single_prompt_input_name=args.single_prompt_input,
        character_lora_strength=args.character_lora_strength,
        lora_1_enabled=resolved["lora_1_enabled"],
        lora_1_name=resolved["lora_1_name"],
        lora_1_strength_model=resolved["lora_1_strength_model"],
        lora_1_strength_clip=resolved["lora_1_strength_clip"],
        lora_1_strengths_explicit=resolved["lora_1_strengths_explicit"],
        randomize_seed=args.randomize_seed,
        seed_offset=args.seed_offset,
        segment_length_mode=args.segment_length_mode,
        min_duration=resolved["min_duration"],
        max_duration=resolved["max_duration"],
        allow_out_of_range_clips=args.allow_out_of_range_clips,
        debug_workflows_dir=args.debug_workflows_dir,
        preroll_frames=preroll_frames,
        tail_loss_frames=tail_loss_frames,
        round_render_frames_to_8n1=round_render_frames_to_8n1,
        postprocess=not args.no_postprocess,
        ffmpeg_path=args.ffmpeg,
        postprocess_reencode=not args.postprocess_streamcopy,
        model_resolver=model_resolver,
    )

    use_case = RenderVideoScenesUseCase(
        backend=backend,
        artifact_store=JsonArtifactStore(),
    )

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

        rendered = use_case.execute(
            RenderVideoScenesRequest(
                render_plan_path=Path(args.render_plan),
                workflow_path=Path(args.workflow),
                audio_file=Path(args.audio),
                storyboard_dir=Path(args.storyboard_dir),
                output_dir=Path(args.output_dir),
                render_mode=args.render_mode,
                single_prompt_workflow_path=Path(args.single_prompt_workflow) if args.single_prompt_workflow else None,
                limit=args.limit,
                scene_numbers=scene_numbers,
                skip_existing=not args.no_skip_existing,
                uploaded_audio_name=args.uploaded_audio_name,
                upload_audio=not args.no_upload_audio,
                upload_startframes=not args.no_upload_startframes,
                anchors=WorkflowAnchorConfig(
                    single_prompt_title=args.single_prompt_title,
                    single_prompt_input=args.single_prompt_input,
                ),
                on_scene_complete=lambda _output, completed, _total: progress.update(
                    task,
                    completed=completed,
                ),
            )
        )
        progress.update(task, completed=len(rendered))

    concat_file = rewrite_concat_list(rendered, args.output_dir)
    console.print(f"[green]âœ“[/green] Rendered/available LTX clips: [yellow]{len(rendered)}[/yellow]")
    console.print(f"[green]âœ“[/green] FFmpeg concat list: [cyan]{concat_file}[/cyan]")
    console.print()
    project_name = resolved["project_config"].project_name if resolved["project_config"] else None
    video_only, final_concat = final_concat_paths(args.output_dir, project_name)
    console.print("Concat + original-audio mux commands:")
    console.print(f'[bold]ffmpeg -y -f concat -safe 0 -i "{concat_file}" -an -c:v copy "{video_only}"[/bold]')
    console.print(f'[bold]ffmpeg -y -i "{video_only}" -i "{args.audio}" -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 320k -shortest "{final_concat}"[/bold]')


if __name__ == "__main__":
    main()
