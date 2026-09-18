"""Scene stage runners for the composition pipeline (M-20, P7 of #1194).

Moved mechanically from ``feverslop.composition.stage_runners``. ``stage_runners`` re-exports these so existing imports and
``patch("...stage_runners.X")`` targets keep working.
"""
from __future__ import annotations

from feverslop.adapters.canonical_plan_store import CanonicalPlanStore
from feverslop.application.render_storyboard import RenderStoryboardRequest
from feverslop.application.sync_project_render_settings import (
    sync_project_render_settings,
)
from feverslop.composition.render_storyboard import build_render_storyboard_use_case
from feverslop.config.app_config import AppConfig
from feverslop.config.project_config import ProjectConfig
from feverslop.tools.storyboard_page import generate_storyboard_page

from ..config_loader import (
    PipelineRunState,
    count_render_plan_items,
)
from .progress import (
    RenderProgressReporter,
    _canonical_plan_path,
    _report,
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
