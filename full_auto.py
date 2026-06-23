from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from feverslop.application.full_auto import FullAutoRequest
from feverslop.composition.full_auto import build_full_auto_use_case
from feverslop.path_utils import coerce_local_path


console = Console()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a full FeverSlop project from an idea and style.")
    parser.add_argument("--idea", required=True)
    parser.add_argument("--style", required=True)
    parser.add_argument("--project-name", default=None)
    parser.add_argument("--projects-dir", default="projects")
    parser.add_argument("--app-config", default="app_config.json")
    parser.add_argument("--workflow", default=str(Path("workflows") / "audio_song.json"))
    parser.add_argument("--duration-seconds", type=float, default=120.0)
    parser.add_argument("--language", default="en")
    parser.add_argument("--bpm", type=int, default=None)
    parser.add_argument("--keyscale", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-video-pipeline", action="store_true")
    parser.add_argument("--concept-batch-size", type=int, default=10)
    parser.add_argument("--storyboard-workflow", default=str(Path("workflows") / "image_t2i_startframe_v1.json"))
    parser.add_argument("--relay-workflow", default="")
    parser.add_argument("--single-prompt-workflow", default=str(Path("workflows") / "video_ltxv_i2v_v1.json"))
    parser.add_argument("--render-mode", choices=["auto", "relay", "single_prompt"], default="single_prompt")
    parser.add_argument("--single-prompt-title", default="#PROMPT")
    parser.add_argument("--single-prompt-input", default="text")
    parser.add_argument("--storyboard-lora-strength", type=float, default=None)
    parser.add_argument("--video-character-lora-strength", type=float, default=None)
    parser.add_argument("--video-lora-1-strength-model", type=float, default=None)
    parser.add_argument("--video-lora-1-strength-clip", type=float, default=None)
    parser.add_argument("--lora-split-enabled", dest="lora_split_enabled", action="store_true", default=None)
    parser.add_argument("--no-lora-split-enabled", dest="lora_split_enabled", action="store_false")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-scene", type=int, default=16)
    parser.add_argument(
        "--rolling-frame-profile",
        choices=["original", "safe", "off"],
        default="original",
    )
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--skip-main-pipeline", action="store_true")
    parser.add_argument("--skip-relay-compact", action="store_true")
    parser.add_argument("--skip-anchor-fix", action="store_true")
    parser.add_argument("--skip-storyboard", action="store_true")
    parser.add_argument("--skip-storyboard-page", action="store_true")
    parser.add_argument("--skip-ltx", action="store_true")
    parser.add_argument("--skip-final-concat", action="store_true")
    parser.add_argument("--diagnostic-original-audio-mux", action="store_true")
    parser.add_argument("--no-original-audio-mux", action="store_true")
    return parser


def request_from_args(args: argparse.Namespace) -> FullAutoRequest:
    return FullAutoRequest(
        idea=args.idea,
        style=args.style,
        project_name=args.project_name,
        projects_dir=Path(args.projects_dir),
        duration_seconds=float(args.duration_seconds),
        language=args.language,
        bpm=args.bpm,
        keyscale=args.keyscale,
        seed=int(args.seed),
        run_video_pipeline=bool(args.run_video_pipeline),
        runner_options={
            "app_config": args.app_config,
            "concept_batch_size": int(args.concept_batch_size),
            "storyboard_workflow": args.storyboard_workflow,
            "relay_workflow": args.relay_workflow,
            "single_prompt_workflow": args.single_prompt_workflow,
            "render_mode": args.render_mode,
            "single_prompt_title": args.single_prompt_title,
            "single_prompt_input": args.single_prompt_input,
            "storyboard_lora_strength": args.storyboard_lora_strength,
            "video_character_lora_strength": args.video_character_lora_strength,
            "video_lora_1_strength_model": args.video_lora_1_strength_model,
            "video_lora_1_strength_clip": args.video_lora_1_strength_clip,
            "lora_split_enabled": args.lora_split_enabled,
            "skip_tests": bool(args.skip_tests),
            "smoke_only": bool(args.smoke_only),
            "smoke_scene": int(args.smoke_scene),
            "rolling_frame_profile": args.rolling_frame_profile,
            "no_skip_existing": bool(args.no_skip_existing),
            "skip_main_pipeline": bool(args.skip_main_pipeline),
            "skip_relay_compact": bool(args.skip_relay_compact),
            "skip_anchor_fix": bool(args.skip_anchor_fix),
            "skip_storyboard": bool(args.skip_storyboard),
            "skip_storyboard_page": bool(args.skip_storyboard_page),
            "skip_ltx": bool(args.skip_ltx),
            "skip_final_concat": bool(args.skip_final_concat),
            "diagnostic_original_audio_mux": bool(args.diagnostic_original_audio_mux),
            "no_original_audio_mux": bool(args.no_original_audio_mux),
        },
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    result = build_full_auto_use_case(
        app_config_path=coerce_local_path(args.app_config),
        workflow_path=coerce_local_path(args.workflow),
    ).execute(request_from_args(args))
    console.print(f"Project config: [cyan]{result.project_config_path}[/cyan]")
    console.print(f"Generated audio: [cyan]{result.audio_path}[/cyan]")
    if result.final_video_path:
        console.print(f"Final video: [cyan]{result.final_video_path}[/cyan]")


if __name__ == "__main__":
    main()
