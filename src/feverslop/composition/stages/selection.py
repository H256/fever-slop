"""Stage selection, registry, and render-plan bootstrap (M-20, P6 of #1194).

Moved mechanically from ``feverslop.composition.stage_runners``. ``stage_runners`` re-exports these so existing imports and
``patch("...stage_runners.X")`` targets keep working.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.config.project_config import ProjectConfig

from ..arg_parser import PipelineStage
from ..config_loader import PipelineRunContext, runner_root
from .final_stages import (
    _run_concat_video_only_stage,
    _run_diagnostic_scene_audio_concat_stage,
    _run_facefix_concat_stage,
    _run_facefix_stage,
    _run_mux_original_audio_stage,
    _run_timeline_export_stage,
    _run_upscale_stage,
)
from .msr_stages import (
    _run_ingredients_sheets_stage,
    _run_msr_prompt_enrich_stage,
    _run_msr_reference_sheets_stage,
    _run_msr_references_stage,
)
from .plan_stages import (
    _run_anchor_fix_stage,
    _run_h3_prompts_stage,
    _run_main_pipeline_stage,
    _run_relay_compact_stage,
    _run_render_plan_stage,
    _run_tests_stage,
)
from .progress import _report
from .render_stages import (
    _run_ltx_prepare_workflows_stage,
    _run_ltx_render_scenes_stage,
)
from .scenes import (
    _run_set_resolution_stage,
    _run_sync_project_settings_stage,
    _run_storyboard_frames_stage,
    _run_storyboard_page_stage,
)


STAGE_RUNNERS = {
    PipelineStage.TESTS: _run_tests_stage,
    PipelineStage.MAIN_PIPELINE: _run_main_pipeline_stage,
    PipelineStage.H3_PROMPTS: _run_h3_prompts_stage,
    PipelineStage.RENDER_PLAN: _run_render_plan_stage,
    PipelineStage.RELAY_COMPACT: _run_relay_compact_stage,
    PipelineStage.ANCHOR_FIX: _run_anchor_fix_stage,
    PipelineStage.SET_RESOLUTION: _run_set_resolution_stage,
    PipelineStage.SYNC_PROJECT_SETTINGS: _run_sync_project_settings_stage,
    PipelineStage.STORYBOARD_FRAMES: _run_storyboard_frames_stage,
    PipelineStage.STORYBOARD_PAGE: _run_storyboard_page_stage,
    PipelineStage.MSR_REFERENCES: _run_msr_references_stage,
    PipelineStage.MSR_REFERENCE_SHEETS: _run_msr_reference_sheets_stage,
    PipelineStage.MSR_PROMPT_ENRICH: _run_msr_prompt_enrich_stage,
    PipelineStage.INGREDIENTS_SHEETS: _run_ingredients_sheets_stage,
    PipelineStage.LTX_PREPARE_WORKFLOWS: _run_ltx_prepare_workflows_stage,
    PipelineStage.LTX_RENDER_SCENES: _run_ltx_render_scenes_stage,
    PipelineStage.PREPARE_WORKFLOWS: _run_ltx_prepare_workflows_stage,
    PipelineStage.RENDER_SCENES: _run_ltx_render_scenes_stage,
    PipelineStage.UPSCALE: _run_upscale_stage,
    PipelineStage.CONCAT_VIDEO_ONLY: _run_concat_video_only_stage,
    PipelineStage.MUX_ORIGINAL_AUDIO: _run_mux_original_audio_stage,
    PipelineStage.DIAGNOSTIC_SCENE_AUDIO_CONCAT: _run_diagnostic_scene_audio_concat_stage,
    PipelineStage.FACEFIX: _run_facefix_stage,
    PipelineStage.FACEFIX_CONCAT: _run_facefix_concat_stage,
    PipelineStage.EXPORT_TIMELINE: _run_timeline_export_stage,
    PipelineStage.OPENSHOT_EXPORT: _run_timeline_export_stage,
}

STAGE_LABELS = {
    PipelineStage.TESTS: "tests",
    PipelineStage.MAIN_PIPELINE: "Main pipeline",
    PipelineStage.H3_PROMPTS: "H3 prompts",
    PipelineStage.RENDER_PLAN: "Render plan",
    PipelineStage.RELAY_COMPACT: "relay compact",
    PipelineStage.ANCHOR_FIX: "anchor fix",
    PipelineStage.SET_RESOLUTION: "Set resolution",
    PipelineStage.SYNC_PROJECT_SETTINGS: "Sync project render settings",
    PipelineStage.STORYBOARD_FRAMES: "Storyboard frames",
    PipelineStage.STORYBOARD_PAGE: "Storyboard page",
    PipelineStage.REFERENCE_RENDER: "Reference render",
    PipelineStage.REFERENCE_SHEETS: "Reference sheets",
    PipelineStage.MSR_REFERENCES: "MSR references",
    PipelineStage.MSR_REFERENCE_SHEETS: "MSR reference sheets",
    PipelineStage.MSR_PROMPT_ENRICH: "MSR prompt enrichment",
    PipelineStage.INGREDIENTS_SHEETS: "Ingredients scene sheets",
    PipelineStage.LTX_PREPARE_WORKFLOWS: "Prepare LTX workflows",
    PipelineStage.LTX_RENDER_SCENES: "Video render",
    PipelineStage.PREPARE_WORKFLOWS: "Prepare workflows",
    PipelineStage.RENDER_SCENES: "Render scenes",
    PipelineStage.UPSCALE: "SeedVR2 upscale",
    PipelineStage.CONCAT_VIDEO_ONLY: "Final concat video-only",
    PipelineStage.MUX_ORIGINAL_AUDIO: "Mux original audio",
    PipelineStage.DIAGNOSTIC_SCENE_AUDIO_CONCAT: "Diagnostic scene-audio concat",
    PipelineStage.FACEFIX: "FaceFix postprocessing",
    PipelineStage.FACEFIX_CONCAT: "FaceFix final concat",
    PipelineStage.EXPORT_TIMELINE: "Timeline export",
    PipelineStage.OPENSHOT_EXPORT: "OpenShot project export",
}


def run_unittest_suite() -> None:
    subprocess.run(["uv", "run", "python", "-m", "unittest", "discover", "-s", "tests"], check=True, cwd=runner_root())


def write_step(message: str) -> None:
    _report()
    _report(f"==> {message}")


def resolve_pipeline_stages(args: argparse.Namespace) -> list[PipelineStage]:
    selected = getattr(args, "stages", None)
    if selected:
        aliases = {
            PipelineStage.REFERENCE_RENDER.value: PipelineStage.MSR_REFERENCES,
            PipelineStage.REFERENCE_SHEETS.value: PipelineStage.MSR_REFERENCE_SHEETS,
            PipelineStage.PREPARE_WORKFLOWS.value: PipelineStage.LTX_PREPARE_WORKFLOWS,
            PipelineStage.RENDER_SCENES.value: PipelineStage.LTX_RENDER_SCENES,
        }
        return [aliases.get(stage, PipelineStage(stage)) for stage in selected]

    # --set-resolution is a special mode: just update config + render plan + re-prepare
    set_res = getattr(args, "set_resolution", None)
    if set_res is not None:
        return [PipelineStage.SET_RESOLUTION]

    stages: list[PipelineStage] = []
    if not args.skip_tests:
        stages.append(PipelineStage.TESTS)
    if not args.skip_main_pipeline:
        stages.append(PipelineStage.MAIN_PIPELINE)
    else:
        _report("Skipping main pipeline; using existing timeline, prompts, and render plan.")
    if not args.skip_relay_compact and args.render_mode != "single_prompt":
        stages.append(PipelineStage.RELAY_COMPACT)
    if not args.skip_anchor_fix:
        stages.append(PipelineStage.ANCHOR_FIX)
    if args.video_pipeline in ("ltx_msr", "minimax-h3-r2v"):
        if not args.skip_msr_reference_render:
            stages.append(PipelineStage.MSR_REFERENCES)
        else:
            _report("Skipping MSR reference rendering; using existing reference manifests.")
        stages.append(PipelineStage.MSR_REFERENCE_SHEETS)
        if args.video_pipeline == "minimax-h3-r2v":
            stages.append(PipelineStage.H3_PROMPTS)
            stages.append(PipelineStage.RENDER_PLAN)
        if args.video_pipeline == "ltx_msr" and not args.skip_msr_prompt_enrichment:
            stages.append(PipelineStage.MSR_PROMPT_ENRICH)
        elif args.video_pipeline == "ltx_msr":
            _report("Skipping MSR prompt enrichment; using existing MSR prompt fields.")
    elif args.video_pipeline == "ltx_ingredients":
        if not args.skip_msr_reference_render:
            stages.append(PipelineStage.MSR_REFERENCES)
        else:
            _report("Skipping MSR reference rendering; using existing reference manifests.")
        stages.append(PipelineStage.MSR_REFERENCE_SHEETS)
        if not args.skip_msr_prompt_enrichment:
            stages.append(PipelineStage.MSR_PROMPT_ENRICH)
        else:
            _report("Skipping MSR prompt enrichment; using existing MSR prompt fields.")
        if not getattr(args, "skip_ingredients_sheets", False):
            stages.append(PipelineStage.INGREDIENTS_SHEETS)
        else:
            _report("Skipping Ingredients sheets; using existing sheets or references.")
    else:
        if not args.skip_storyboard:
            stages.append(PipelineStage.STORYBOARD_FRAMES)
        if not args.skip_storyboard_page:
            stages.append(PipelineStage.STORYBOARD_PAGE)
    if not args.skip_ltx:
        if args.video_pipeline in ("ltx_msr", "ltx_ingredients"):
            stages.append(PipelineStage.LTX_PREPARE_WORKFLOWS)
        stages.append(PipelineStage.LTX_RENDER_SCENES)
    if not args.skip_final_concat:
        config_path = getattr(args, "project_config", None)
        if config_path is None and getattr(args, "project_root", None):
            config_path = Path(args.project_root) / "config.json"
        config_upscale_enabled = False
        if config_path and Path(config_path).is_file():
            config_upscale_enabled = ProjectConfig.load(config_path).upscale.enabled
        upscale_enabled = bool(getattr(args, "upscale", False)) or config_upscale_enabled
        if not args.skip_facefix:
            stages.append(PipelineStage.FACEFIX)
        else:
            _report("Skipping FaceFix postprocessing.")
        if upscale_enabled:
            stages.append(PipelineStage.UPSCALE)
        stages.append(PipelineStage.CONCAT_VIDEO_ONLY)
        stages.append(PipelineStage.MUX_ORIGINAL_AUDIO)
        if args.diagnostic_original_audio_mux:
            stages.append(PipelineStage.DIAGNOSTIC_SCENE_AUDIO_CONCAT)
        elif args.no_original_audio_mux:
            _report("--no-original-audio-mux is deprecated; original-audio muxing is now always used for final concat.")
        if not getattr(args, "skip_openshot_export", False):
            stages.append(PipelineStage.EXPORT_TIMELINE)
        else:
            _report("Skipping OpenShot project export.")
    elif not args.skip_facefix:
        stages.append(PipelineStage.FACEFIX)
    else:
        _report("Skipping FaceFix postprocessing.")
    return stages


def _initial_render_plan(context: PipelineRunContext, args: argparse.Namespace, stages: list[PipelineStage]) -> Path:
    upstream_stages = {PipelineStage.MAIN_PIPELINE, PipelineStage.RELAY_COMPACT, PipelineStage.ANCHOR_FIX, PipelineStage.MSR_REFERENCE_SHEETS, PipelineStage.INGREDIENTS_SHEETS}
    render_dir = context.render_dir
    legacy_base = render_dir / f"render_plan_{context.song_id}.json"
    legacy_references = render_dir / f"render_plan_{context.song_id}_refs.json"
    legacy_ingredients = render_dir / f"render_plan_{context.song_id}_ingredients.json"
    ingredients_projection_stages = {
        PipelineStage.MSR_PROMPT_ENRICH,
        PipelineStage.INGREDIENTS_SHEETS,
    }
    if (
        args.video_pipeline == "ltx_ingredients"
        and ingredients_projection_stages.intersection(stages)
        and PipelineStage.MSR_REFERENCE_SHEETS not in stages
    ):
        existing = context.artifact_layout.find_plan(
            context.reference_plan,
            legacy_paths=[legacy_references],
        )
        if existing:
            return existing
    if stages in ([PipelineStage.OPENSHOT_EXPORT], [PipelineStage.EXPORT_TIMELINE]):
        for plan_path, legacy_paths in (
            (context.render_plan, [legacy_base]),
            (context.reference_plan, [legacy_references]),
            (context.ingredients_plan, [legacy_ingredients]),
            (context.anchored_plan, []),
            (context.compact_plan, []),
        ):
            existing = context.artifact_layout.find_plan(plan_path, legacy_paths=legacy_paths)
            if existing:
                return existing
        return context.render_plan
    if args.video_pipeline == "ltx_msr" and not upstream_stages.intersection(stages):
        existing = context.artifact_layout.find_plan(context.reference_plan, legacy_paths=[legacy_references])
        if existing:
            return existing
    if args.video_pipeline == "ltx_ingredients" and not upstream_stages.intersection(stages):
        existing = context.artifact_layout.find_plan(context.ingredients_plan, legacy_paths=[legacy_ingredients])
        if existing:
            return existing
    # MiniMax R2V needs reference paths (actor_msr_paths, location_msr_path) from
    # an existing MSR/ingredients plan so it can patch them into its workflow.
    if args.video_pipeline == "minimax-h3-r2v":
        if context.render_plan.is_file():
            scenes = JsonArtifactStore().read_json(context.render_plan)
            if scenes and all((scene.get("h3") or {}).get("prompt") for scene in scenes):
                return context.render_plan
        for plan_path, legacy_paths in (
            (context.ingredients_plan, [legacy_ingredients]),
            (context.reference_plan, [legacy_references]),
            (context.render_plan, [legacy_base]),
        ):
            existing = context.artifact_layout.find_plan(plan_path, legacy_paths=legacy_paths)
            if existing:
                return existing
        return context.render_plan
    return context.artifact_layout.find_plan(context.render_plan, legacy_paths=[legacy_base]) or context.render_plan
