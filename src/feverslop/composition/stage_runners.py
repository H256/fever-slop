from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.adapters.canonical_plan_store import CanonicalPlanStore
from feverslop.application.render_storyboard import RenderStoryboardRequest
from feverslop.application.sync_project_render_settings import (
    sync_project_render_settings,
)
from feverslop.composition.generate_render_plan import (
    build_generate_render_plan_use_case,  # noqa: F401
    )
from feverslop.composition.render_storyboard import build_render_storyboard_use_case
from feverslop.config.app_config import AppConfig
from feverslop.config.project_config import ProjectConfig
from feverslop.tools.storyboard_page import generate_storyboard_page

from .arg_parser import PipelineStage
from .config_loader import (
    PipelineRunContext,
    PipelineRunState,
    count_render_plan_items,
    runner_root,
)
# Plan-stage runners now live in feverslop.composition.stages.plan_stages (M-20, P2).
# Re-exported here so existing imports and patch("...stage_runners.X") targets keep working.
from .stages.plan_stages import (  # noqa: F401
    _get_reference_bible_parser,
    _get_resolution,
    _selected_video_workflows,
    _run_tests_stage,
    _run_main_pipeline_stage,
    _read_h3_input,
    _select_pipeline_scenes,
    _report_reference_fallbacks,
    _seed_reference_bindings,
    _discover_stem_files,
    _merge_reference_paths_into_h3_segments,
    _select_h3_segments,
    _run_h3_prompts_stage,
    _run_render_plan_stage,
    _preserve_enriched_reference_paths,
    _run_relay_compact_stage,
    _run_anchor_fix_stage,
)

# Progress-reporting state and helpers now live in
# feverslop.composition.stages.progress (M-20). Re-exported here so existing
# ``import`` and ``patch("...stage_runners.X")`` targets keep working.
from .stages.progress import (  # noqa: F401
    VIDEO_SCENE_PROGRESS_LABEL,
    RenderProgressReporter,
    _canonical_plan_path,
    _report,
    _scene_progress_callback,
    console,
    get_reporter,
    set_reporter,
)

# MSR/Ingredients stage runners now live in
# feverslop.composition.stages.msr_stages (M-20, P3). Re-exported here so
# existing imports and patch("...stage_runners.X") targets keep working.
from .stages.msr_stages import (  # noqa: F401
    OpenAICompatibleLLMClient,
    _run_ingredients_sheets_stage,
    _run_msr_prompt_enrich_stage,
    _run_msr_reference_sheets_stage,
    _run_msr_references_stage,
    enrich_render_plan_with_msr_prompts,
    enrich_render_plan_with_reference_sheets,
    reference_manifests_reusable,
    render_reference_bible,
)

# Render-stage runners (LTX prepare, render scenes, continuity) now live in
# feverslop.composition.stages.render_stages (M-20, P4). Re-exported here so
# existing imports and patch("...stage_runners.X") targets keep working.
from .stages.render_stages import (  # noqa: F401
    _PROFILE_UNSET,
    _all_render_scenes,
    _attach_music_continuity_handoffs,
    _canonical_dependencies_from_scene,
    _continuation_boundary_manifest_valid,
    _continuity_downstream,
    _load_continuity_dirty,
    _missing_prepare_inputs,
    _music_handoff_predecessors,
    _music_handoff_prompt,
    _prepared_scene_is_fresh,
    _project_visual_consistency_contracts,
    _require_fresh_reference_projections,
    _resolved_startframe_profile,
    _run_ltx_prepare_workflows_stage,
    _run_ltx_render_scenes_stage,
    _run_visual_consistency_preflight,
    _selected_render_scenes,
    _select_render_scenes,
    _specialized_video_use_case,
    _stored_consistency_contract,
    _write_continuity_dirty,
)


# Final-assembly stage runners (cutless/concat/mux/upscale/diagnostic/timeline/facefix)
# now live in feverslop.composition.stages.final_stages (M-20, P5). Re-exported
# here so existing imports and patch("...stage_runners.X") targets keep working.
from .stages.final_stages import (  # noqa: F401
    _run_concat_video_only_stage,
    _run_diagnostic_scene_audio_concat_stage,
    _run_facefix_concat_stage,
    _run_facefix_stage,
    _run_mux_original_audio_stage,
    _run_timeline_export_stage,
    _run_upscale_stage,
    _assemble_declared_cutless_groups,
    _stage_final_postprocessor,
)

def _run_set_resolution_stage(state: PipelineRunState) -> None:
    """Persist resolution to config and the canonical plan for safe resume."""
    set_res = getattr(state.args, "set_resolution", None)
    if set_res is None:
        raise ValueError("--set-resolution WxH is required for the set_resolution stage")

    width = set_res.width
    height = set_res.height
    megapixels = set_res.megapixels
    label = f"{width}x{height}" if megapixels is None else f"{megapixels} MP"
    _report(f"Setting resolution to {label}...")

    # 1. Patch config.json
    ProjectConfig.set_resolution_on_disk(
        state.context.project_config_path,
        width=width,
        height=height,
        megapixels=megapixels,
    )
    _report(f"[green]Updated config.json resolution to {label}[/green]")

    # Keep derived plans and manifests untouched. Their old fingerprints are
    # the evidence that makes the next normal resume rebuild the right work.
    store = CanonicalPlanStore(state.context.project_config_dir)
    snapshot = store.capture_regeneration()
    if not snapshot.exists:
        _report(
            "[green]Resolution saved. The canonical plan will receive it when "
            "the normal pipeline creates the plan.[/green]",
        )
        return
    updated = []
    for scene in snapshot.scenes:
        patched = dict(scene)
        if width is not None:
            patched["width"] = width
        if height is not None:
            patched["height"] = height
        if megapixels is not None:
            patched["megapixels"] = megapixels
        updated.append(patched)
    store.commit_regeneration(snapshot, updated)
    state.plan_for_next_step = state.context.artifact_layout.base_plan
    _report(
        "[green]Updated canonical resolution. Use the normal --dry-run/--resume "
        "pair to rebuild stale workflows and clips.[/green]",
    )


def _run_sync_project_settings_stage(state: PipelineRunState) -> None:
    settings = getattr(state.args, "project_render_settings", None)
    if settings is None:
        raise ValueError("sync_project_settings requires resolved project render settings")
    changed = sync_project_render_settings(
        CanonicalPlanStore(state.context.project_config_dir),
        settings,
    )
    state.plan_for_next_step = state.context.artifact_layout.base_plan
    if changed:
        _report("[green]Canonical project render settings synchronized.[/green]")
    else:
        _report("[dim]Canonical project render settings already match.[/dim]")


def _run_storyboard_frames_stage(state: PipelineRunState) -> None:
    if state.args.video_pipeline == "ltx_msr":
        raise ValueError("storyboard_frames is not used by ltx_msr")
    app_config = AppConfig.load(state.app_config_path, required_keys=["llm", "comfyui"])
    storyboard_use_case = build_render_storyboard_use_case(
        app_config=app_config,
        workflow_path=state.storyboard_workflow,
        output_dir=state.context.storyboard_dir,
    )
    storyboard_total = count_render_plan_items(state.plan_for_next_step)
    with RenderProgressReporter("Rendering storyboard frames", storyboard_total) as storyboard_progress:
        storyboard_use_case.execute(
            RenderStoryboardRequest(
                render_plan_path=state.plan_for_next_step,
                canonical_plan_path=_canonical_plan_path(state),
                workflow_path=state.storyboard_workflow,
                output_dir=state.context.storyboard_dir,
                character_lora_strength=state.args.storyboard_lora_strength,
                on_frame_complete=storyboard_progress.update,
            ),
        )


def _run_storyboard_page_stage(state: PipelineRunState) -> None:
    if state.args.video_pipeline == "ltx_msr":
        raise ValueError("storyboard_page is not used by ltx_msr")
    generate_storyboard_page(
        render_plan_path=state.plan_for_next_step,
        storyboard_dir=state.context.storyboard_dir,
        output_html=state.context.storyboard_page,
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
            PipelineStage.EXPORT_TIMELINE.value: PipelineStage.EXPORT_TIMELINE,
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
