from __future__ import annotations

import argparse
from pathlib import Path


RUNNER_ARGUMENTS = (
    ("app_config", ("--app-config",), {"default": "app_config.json"}),
    ("concept_batch_size", ("--concept-batch-size",), {"type": int, "default": 10}),
    ("storyboard_workflow", ("--storyboard-workflow",), {"default": str(Path("workflows") / "image_t2i_startframe_v1.json")}),
    ("reference_hero_workflow", ("--reference-hero-workflow",), {"default": str(Path("workflows") / "image_t2i_startframe_krea_v1.json")}),
    ("reference_edit_workflow", ("--reference-edit-workflow",), {"default": str(Path("workflows") / "image_edit_flux2_klein_1ref_v1.json")}),
    ("msr_workflow", ("--msr-workflow",), {"default": str(Path("workflows") / "video_ltxv_msr_1actor_1background_v2.json")}),
    ("ingredients_workflow", ("--ingredients-workflow",), {"default": str(Path("workflows") / "video_ltxv_ingredients_audio_2stage_v2.json")}),
    ("relay_workflow", ("--relay-workflow",), {"default": ""}),
    ("single_prompt_workflow", ("--single-prompt-workflow",), {"default": str(Path("workflows") / "video_ltxv_i2v_v1.json")}),
    ("video_pipeline", ("--video-pipeline",), {"choices": ["ltx_i2v", "ltx_msr", "ltx_ingredients"], "default": "ltx_i2v"}),
    ("render_mode", ("--render-mode",), {"choices": ["auto", "relay", "single_prompt"], "default": "single_prompt"}),
    ("single_prompt_title", ("--single-prompt-title",), {"default": "#PROMPT"}),
    ("single_prompt_input", ("--single-prompt-input",), {"default": "text"}),
    ("rolling_frame_profile", ("--rolling-frame-profile",), {"choices": ["original", "safe", "off"], "default": "original"}),
    ("storyboard_lora_strength", ("--storyboard-lora-strength",), {"type": float, "default": None}),
    ("video_character_lora_strength", ("--video-character-lora-strength",), {"type": float, "default": None}),
    ("video_lora_1_strength_model", ("--video-lora-1-strength-model",), {"type": float, "default": None}),
    ("video_lora_1_strength_clip", ("--video-lora-1-strength-clip",), {"type": float, "default": None}),
    ("lora_split_enabled", ("--lora-split-enabled",), {"dest": "lora_split_enabled", "action": "store_true", "default": None}),
    ("lora_split_enabled", ("--no-lora-split-enabled",), {"dest": "lora_split_enabled", "action": "store_false"}),
    ("randomize_seed", ("--randomize-seed",), {"action": "store_true"}),
    ("scenes", ("--scenes",), {"default": None, "help": "Render only selected scenes, for example 3,5-8,15."}),
    ("smoke_scene", ("--smoke-scene",), {"type": int, "default": 16}),
    ("smoke_only", ("--smoke-only",), {"action": "store_true"}),
    ("no_skip_existing", ("--no-skip-existing",), {"action": "store_true"}),
    ("skip_tests", ("--skip-tests",), {"action": "store_true"}),
    ("skip_main_pipeline", ("--skip-main-pipeline",), {"action": "store_true"}),
    ("skip_relay_compact", ("--skip-relay-compact",), {"action": "store_true"}),
    ("skip_anchor_fix", ("--skip-anchor-fix",), {"action": "store_true"}),
    ("skip_storyboard", ("--skip-storyboard",), {"action": "store_true"}),
    ("skip_storyboard_page", ("--skip-storyboard-page",), {"action": "store_true"}),
    ("skip_msr_reference_render", ("--skip-msr-reference-render",), {"action": "store_true"}),
    ("skip_msr_prompt_enrichment", ("--skip-msr-prompt-enrichment",), {"action": "store_true"}),
    ("skip_ingredients_sheets", ("--skip-ingredients-sheets",), {"action": "store_true"}),
    ("skip_ltx", ("--skip-ltx",), {"action": "store_true"}),
    ("skip_final_concat", ("--skip-final-concat",), {"action": "store_true"}),
    ("diagnostic_original_audio_mux", ("--diagnostic-original-audio-mux",), {"action": "store_true"}),
    ("no_original_audio_mux", ("--no-original-audio-mux",), {"action": "store_true"}),
)


def add_runner_options(parser: argparse.ArgumentParser) -> None:
    for _name, flags, kwargs in RUNNER_ARGUMENTS:
        parser.add_argument(*flags, **kwargs)


def runner_options_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {name: getattr(args, name) for name, _flags, _kwargs in RUNNER_ARGUMENTS}


def build_runner_argv(project_config_path: Path, options: dict[str, object]) -> list[str]:
    argv = ["--project-config", str(project_config_path)]
    for name, flags, kwargs in RUNNER_ARGUMENTS:
        value = options.get(name)
        action = kwargs.get("action")
        if action == "store_true" and value is True:
            argv.append(flags[0])
        elif action == "store_false" and value is False:
            argv.append(flags[0])
        elif action is None and value not in (None, ""):
            argv.extend([flags[0], str(value)])
    return argv
