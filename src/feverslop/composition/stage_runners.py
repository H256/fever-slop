from __future__ import annotations

# Patch-target kept re-exported: test_run_pipeline patches
# ``stage_runners.build_generate_render_plan_use_case`` (guard that the skipped
# main-pipeline stage never builds a render plan). The real lookup happens in
# feverslop.composition.generate_render_plan / stages.plan_stages.
from feverslop.composition.generate_render_plan import (
    build_generate_render_plan_use_case,  # noqa: F401
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

# Scene stage runners (set_resolution, sync_project_settings, storyboard_frames,
# storyboard_page) now live in feverslop.composition.stages.scenes (M-20, P7).
# Re-exported here so existing imports and patch("...stage_runners.X") targets keep working.
from .stages.scenes import (  # noqa: F401
    _run_set_resolution_stage,
    _run_sync_project_settings_stage,
    _run_storyboard_frames_stage,
    _run_storyboard_page_stage,
)

# Stage selection, registry, and render-plan bootstrap now live in
# feverslop.composition.stages.selection (M-20, P6). Re-exported here so
# existing imports and patch("...stage_runners.X") targets keep working.
from .stages.selection import (  # noqa: F401
    STAGE_LABELS,
    STAGE_RUNNERS,
    _initial_render_plan,
    resolve_pipeline_stages,
    run_unittest_suite,
    write_step,
)
