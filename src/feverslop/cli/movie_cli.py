from __future__ import annotations

import argparse
from typing import Any


def _parse_scene_numbers(value: str) -> list[int]:
    try:
        numbers = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--scenes must be comma-separated integers") from exc
    if any(number < 1 for number in numbers):
        raise argparse.ArgumentTypeError("--scenes values must be positive")
    return numbers


def build_movie_arg_parser() -> argparse.ArgumentParser:
    """Build argparse parser for the movie pipeline subcommand."""
    parser = argparse.ArgumentParser(
        description="Run movie pipeline stages for an existing FeverSlop movie project.",
    )
    parser.add_argument("project_dir", help="Movie project directory, for example projects/tm3")
    parser.add_argument("--app-config", default="app_config.json")
    parser.add_argument("--reference-backend", choices=["comfyui", "local"], default=None)
    parser.add_argument("--render-backend", choices=["comfyui", "local"], default=None)
    parser.add_argument("--hero-workflow", default=None)
    parser.add_argument("--edit-workflow", default=None)
    parser.add_argument("--director-workflow", default=None)
    parser.add_argument("--startframe-director-backend", choices=["krea2", "ideogram"], default=None)
    parser.add_argument("--mask-workflow", default=None)
    parser.add_argument("--identity-repair-workflow", default=None)
    parser.add_argument("--detail-workflow", default=None)
    parser.add_argument("--startframe-comfyui-base-url", default=None)
    parser.add_argument("--startframe-validator-base-url", default=None)
    parser.add_argument("--startframe-validator-model", default=None)
    parser.add_argument("--msr-workflow", default=None)
    parser.add_argument("--msr-i2v-workflow", default=None)
    parser.add_argument("--i2v-workflow", default=None)
    parser.add_argument("--r2v-workflow", default=None)
    parser.add_argument("--t2v-workflow", default=None)
    parser.add_argument("--ingredients-workflow", default=None)
    parser.add_argument("--skip-movie-bible", action="store_true", help="Reuse existing movie/bible.json.")
    parser.add_argument("--force-movie-bible", action="store_true", help="Regenerate movie/bible.json from the configured movie planner.")
    parser.add_argument("--movie-planner-backend", choices=["llm", "deterministic", "local"], default=None)
    parser.add_argument("--skip-movie-story-design", action="store_true", help="Reuse existing movie/story_design.json.")
    parser.add_argument("--force-movie-story-design", action="store_true", help="Regenerate movie/story_design.json from project source/render plan.")
    parser.add_argument("--skip-movie-screenplay", action="store_true", help="Reuse existing movie/screenplay.json.")
    parser.add_argument("--force-movie-screenplay", action="store_true", help="Regenerate movie/screenplay.json from project source/render plan.")
    parser.add_argument("--skip-movie-narrative", action="store_true", help="Reuse existing movie/narrative_plan.json.")
    parser.add_argument("--skip-movie-scene-cards", action="store_true", help="Reuse existing movie/scene_cards.json.")
    parser.add_argument("--skip-movie-shot-cards", action="store_true", help="Reuse existing movie/shot_cards.json.")
    parser.add_argument("--skip-movie-continuity", action="store_true", help="Reuse existing movie/continuity_plan.json.")
    parser.add_argument("--skip-movie-plan", action="store_true", help="Reuse existing movie/render_plan.json.")
    parser.add_argument("--skip-movie-references", action="store_true", help="Reuse existing movie reference manifest paths.")
    parser.add_argument("--skip-movie-msr-enrich", action="store_true", help="Reuse existing movie/render_plan_msr.json or render the plain plan.")
    parser.add_argument("--skip-movie-ingredients-sheets", action="store_true", help="Skip Ingredients scene sheet composition.")
    parser.add_argument("--skip-movie-render", action="store_true", help="Stop after syncing/rendering movie references.")
    parser.add_argument("--force-movie-references", action="store_true", help="Render movie references even when manifest paths already exist.")
    parser.add_argument("--keyframe-mode", choices=["none", "start", "start-end"], default="none")
    parser.add_argument("--movie-video-workflow", choices=["msr", "msr-i2v-startframe", "i2v-edit", "startframe-director", "ingredients", "minimax-h3-r2v", "minimax-h3-t2v", "minimax-h3-i2v"], default="msr")
    parser.add_argument("--continuity-keyframes", choices=["none", "last-to-start"], default="none")
    parser.add_argument("--scenes", type=_parse_scene_numbers, default=[], help="Comma-separated scene numbers to prepare or render.")
    parser.add_argument(
        "--write-debug-workflows", action="store_true",
        help="Deprecated alias: prepare canonical movie scene workflows without queueing ComfyUI.",
    )
    parser.add_argument("--debug-workflows-dir", default=None, help="Deprecated compatibility option; canonical scene paths are always used.")
    return parser


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Convert movie CLI args into runtime config dict.

    This is wiring logic that bridges CLI args to the movie pipeline config.
    """
    from feverslop.composition.movie_pipeline_jobs import movie_runtime_config

    config: dict[str, Any] = {}
    for key in (
        "reference_backend",
        "render_backend",
        "hero_workflow",
        "edit_workflow",
        "director_workflow",
        "startframe_director_backend",
        "mask_workflow",
        "identity_repair_workflow",
        "detail_workflow",
        "startframe_comfyui_base_url",
        "startframe_validator_base_url",
        "startframe_validator_model",
        "msr_workflow",
        "msr_i2v_workflow",
        "i2v_workflow",
        "r2v_workflow",
        "t2v_workflow",
        "ingredients_workflow",
        "movie_video_workflow",
        "keyframe_mode",
        "continuity_keyframes",
    ):
        value = getattr(args, key, None)
        if value:
            config[key] = value
    if (
        config.get("movie_video_workflow") == "msr-i2v-startframe"
        and config.get("continuity_keyframes") == "last-to-start"
        and _looks_like_i2v_workflow(config.get("msr_workflow"))
        and not config.get("msr_i2v_workflow")
    ):
        config["msr_i2v_workflow"] = config.pop("msr_workflow")
    return movie_runtime_config(config)


def _looks_like_i2v_workflow(value: object) -> bool:
    from pathlib import Path
    return "i2v" in Path(str(value or "")).name.lower()
