"""Canonical owner of the pipeline-stage vocabulary.

The stage vocabulary is domain language: execution planning, resource
scheduling, continuity, state persistence and CLI selection all talk in
stages. In a hexagonal architecture that vocabulary belongs inside the
hexagon, so this module is the single owner. Composition and other
layers import from here (inward) instead of re-declaring the vocabulary
as string literals.
"""

from __future__ import annotations

from enum import Enum


class PipelineStage(str, Enum):
    TESTS = "tests"
    MAIN_PIPELINE = "main_pipeline"
    H3_PROMPTS = "h3_prompts"
    RENDER_PLAN = "render_plan"
    RELAY_COMPACT = "relay_compact"
    ANCHOR_FIX = "anchor_fix"
    SET_RESOLUTION = "set_resolution"
    SYNC_PROJECT_SETTINGS = "sync_project_settings"
    STORYBOARD_FRAMES = "storyboard_frames"
    STORYBOARD_PAGE = "storyboard_page"
    REFERENCE_RENDER = "reference_render"
    REFERENCE_SHEETS = "reference_sheets"
    MSR_REFERENCES = "msr_references"
    MSR_REFERENCE_SHEETS = "msr_reference_sheets"
    MSR_PROMPT_ENRICH = "msr_prompt_enrich"
    INGREDIENTS_SHEETS = "ingredients_sheets"
    LTX_PREPARE_WORKFLOWS = "ltx_prepare_workflows"
    LTX_RENDER_SCENES = "ltx_render_scenes"
    PREPARE_WORKFLOWS = "prepare_workflows"
    RENDER_SCENES = "render_scenes"
    UPSCALE = "upscale"
    CONCAT_VIDEO_ONLY = "concat_video_only"
    MUX_ORIGINAL_AUDIO = "mux_original_audio"
    DIAGNOSTIC_SCENE_AUDIO_CONCAT = "diagnostic_scene_audio_concat"
    FACEFIX = "facefix"
    FACEFIX_CONCAT = "facefix_concat"
    EXPORT_TIMELINE = "export_timeline"
    # Backward-compatible name for the original automatic OpenShot stage.
    OPENSHOT_EXPORT = "openshot_export"


# Resume ordering: the safe-resume subset, in execution order.
RESUME_STAGE_ORDER = (
    "tests",
    "main_pipeline",
    "sync_project_settings",
    "relay_compact",
    "anchor_fix",
    "msr_references",
    "msr_reference_sheets",
    "h3_prompts",
    "render_plan",
    "msr_prompt_enrich",
    "ingredients_sheets",
    "ltx_prepare_workflows",
    "ltx_render_scenes",
    "facefix",
    "upscale",
    "concat_video_only",
    "mux_original_audio",
    "diagnostic_scene_audio_concat",
    "export_timeline",
)
RESUME_STAGE_INDEX = {stage: index for index, stage in enumerate(RESUME_STAGE_ORDER)}

# Resource classification: which stages own an LLM / ComfyUI resource.
LLM_STAGES = frozenset({
    "main_pipeline",
    "relay_compact",
    "h3_prompts",
    "msr_prompt_enrich",
    "ingredients_sheets",
})
COMFYUI_STAGES = frozenset({
    "storyboard_frames",
    "msr_references",
    "ltx_prepare_workflows",
    "ltx_render_scenes",
    "facefix",
    "upscale",
})
NEUTRAL_STAGES = frozenset({
    "tests",
    "sync_project_settings",
    "anchor_fix",
    "msr_reference_sheets",
    "render_plan",
    "storyboard_page",
    "concat_video_only",
    "mux_original_audio",
    "diagnostic_scene_audio_concat",
    "export_timeline",
})

# Stages that clear the ComfyUI cache / VRAM before they run.
COMFYUI_RENDERING_STAGES = frozenset({
    PipelineStage.STORYBOARD_FRAMES,
    PipelineStage.MSR_REFERENCES,
    PipelineStage.LTX_RENDER_SCENES,
    PipelineStage.RENDER_SCENES,
    PipelineStage.FACEFIX,
    PipelineStage.UPSCALE,
})
