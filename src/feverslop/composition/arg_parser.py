from __future__ import annotations

from enum import Enum
import argparse

from feverslop.adapters.pipeline_runner_options import add_runner_options


class PipelineStage(str, Enum):
    TESTS = "tests"
    MAIN_PIPELINE = "main_pipeline"
    RELAY_COMPACT = "relay_compact"
    ANCHOR_FIX = "anchor_fix"
    STORYBOARD_FRAMES = "storyboard_frames"
    STORYBOARD_PAGE = "storyboard_page"
    MSR_REFERENCES = "msr_references"
    MSR_REFERENCE_SHEETS = "msr_reference_sheets"
    MSR_PROMPT_ENRICH = "msr_prompt_enrich"
    INGREDIENTS_SHEETS = "ingredients_sheets"
    LTX_PREPARE_WORKFLOWS = "ltx_prepare_workflows"
    LTX_RENDER_SCENES = "ltx_render_scenes"
    CONCAT_VIDEO_ONLY = "concat_video_only"
    MUX_ORIGINAL_AUDIO = "mux_original_audio"
    DIAGNOSTIC_SCENE_AUDIO_CONCAT = "diagnostic_scene_audio_concat"
    FACEFIX = "facefix"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FeverSlop pipeline from Python.")
    parser.add_argument("project_root", nargs="?", default=None)
    parser.add_argument("--project-config", default=None)
    parser.add_argument(
        "--stage",
        dest="stages",
        action="append",
        choices=[stage.value for stage in PipelineStage],
        default=None,
        help="Run only a specific atomic pipeline stage. May be passed more than once.",
    )
    add_runner_options(parser)
    return parser
