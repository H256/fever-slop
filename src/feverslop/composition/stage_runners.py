from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.adapters.canonical_plan_store import CanonicalPlanStore
from feverslop.adapters.postprocessor_frame_extractor import (
    PostprocessorFrameExtractor,
)
from feverslop.adapters.prepared_workflow import (
    PreparedWorkflowRenderer,
    WorkflowMaterializationRequest,
    WorkflowMaterializer,
)
from feverslop.adapters.project_visual_consistency import (
    ProjectReferenceManifestAdapter,
    validate_project_scene_artifacts,
)
from feverslop.adapters.reporting import ConsoleReporter
from feverslop.adapters.video_postprocessor import VideoPostProcessor, final_video_postprocessor
from feverslop.adapters.cutless_assembly import CutlessAssemblyService
from feverslop.application.continuity_handoff import ContinuityHandoffUseCase
from feverslop.composition.cutless_assembly import assemble_declared_cutless_group
from feverslop.application.effective_render_plan import (
    CanonicalSceneDependencies,
    project_effective_plan,
)
from feverslop.application.mlt_exporter import export_render_plan_to_mlt
from feverslop.application.openshot_exporter import export_render_plan_to_openshot
from feverslop.application.render_storyboard import RenderStoryboardRequest
from feverslop.application.render_video import (
    RenderVideoScenesRequest,
    patch_render_plan_seed,
)
from feverslop.application.sync_project_render_settings import (
    sync_project_render_settings,
)
from feverslop.application.visual_consistency_preflight import (
    VisualConsistencyPreflightResult,
    preflight_visual_consistency,
    resolve_preflight_workflow_profile,
)
from feverslop.composition.generate_render_plan import (
    build_generate_render_plan_use_case,  # noqa: F401
    )
from feverslop.composition.continuation_scheduler import (
    ContinuationScheduler,
    chains_from_predecessors,
)
from feverslop.composition.render_storyboard import build_render_storyboard_use_case
from feverslop.composition.render_video import (
    RenderVideoCompositionOptions,
    build_render_video_scenes_use_case,
)
from feverslop.config.app_config import AppConfig
from feverslop.config.project_config import ProjectConfig
from feverslop.domain.prepared_workflow import SceneWorkflowManifest
from feverslop.domain.render_plan import RenderPlan, RenderScene
from feverslop.domain.visual_consistency import (
    PreflightMode,
    SceneConsistencyContract,
    can_handoff,
    expand_handoff_selection,
    validate_scene_sequence,
)
from feverslop.ports.rendering import WorkflowAnchorConfig
from feverslop.tools.storyboard_page import generate_storyboard_page, parse_scene_list
from feverslop.utils.io import file_is_valid

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


def _specialized_video_use_case(state: PipelineRunState):
    workflow = _selected_video_workflows(state)[0]
    return build_render_video_scenes_use_case(
        RenderVideoCompositionOptions(
            app_config_path=state.app_config_path,
            project_config_path=state.context.project_config_path,
            render_plan_path=state.plan_for_next_step,
            workflow_path=workflow,
            output_dir=state.context.ltx_dir,
            video_pipeline=state.args.video_pipeline,
            video_workflow_profile=getattr(
                state.args,
                "video_workflow_profile",
                None,
            ),
            randomize_seed=state.args.randomize_seed,
            rolling_frame_profile=state.args.rolling_frame_profile,
            resolution=_get_resolution(state.args),
        ),
        console=console,
    )


def _selected_render_scenes(state: PipelineRunState) -> tuple[RenderScene, ...]:
    return _select_render_scenes(state, _all_render_scenes(state))


def _all_render_scenes(state: PipelineRunState) -> tuple[RenderScene, ...]:
    payload = json.loads(state.plan_for_next_step.read_text(encoding="utf-8-sig"))
    canonical_path = _canonical_plan_path(state)
    canonical_payload = (
        json.loads(canonical_path.read_text(encoding="utf-8-sig"))
        if canonical_path is not None
        else None
    )
    projected = project_effective_plan(payload, canonical_payload)
    _require_fresh_reference_projections(state, payload, projected)
    if not any(scene.get("technical_segment_id") for scene in projected):
        validate_scene_sequence(projected)
    else:
        numbers = [scene.get("scene") for scene in projected]
        if any(type(number) is not int or number <= 0 for number in numbers):
            raise ValueError("Technical render scenes must use positive integer scene IDs")
        if len(numbers) != len(set(numbers)):
            raise ValueError("Technical render scenes must use unique scene IDs")
    return RenderPlan.from_dicts(projected).scenes


def _canonical_dependencies_from_scene(
    scene: dict,
) -> CanonicalSceneDependencies | None:
    projection = scene.get("canonical_projection")
    dependencies = (
        projection.get("dependencies")
        if isinstance(projection, dict)
        else None
    )
    if not isinstance(dependencies, dict):
        return None
    return CanonicalSceneDependencies.from_dict(dependencies)


def _require_fresh_reference_projections(
    state: PipelineRunState,
    stored_scenes: list[dict],
    current_scenes: list[dict],
) -> None:
    for stored, current in zip(stored_scenes, current_scenes, strict=True):
        previous = _canonical_dependencies_from_scene(stored)
        expected = _canonical_dependencies_from_scene(current)
        if (
            previous is None
            or expected is None
            or previous.reference_fingerprint == expected.reference_fingerprint
        ):
            continue
        scene_number = int(current.get("scene") or stored.get("scene") or 0)
        required_stage = (
            "ingredients_sheets"
            if state.args.video_pipeline == "ltx_ingredients"
            else "msr_reference_sheets"
        )
        raise ValueError(
            "Stale derived reference binding from "
            f"{expected.source} for scene {scene_number}: reference binding "
            f"fingerprint changed. Run --stage {required_stage} first.",
        )


def _prepared_scene_is_fresh(
    state: PipelineRunState,
    scene_number: int,
    canonical_dependencies: CanonicalSceneDependencies,
) -> bool:
    workflow_path = state.context.artifact_layout.scene_workflow(scene_number)
    manifest_path = state.context.artifact_layout.scene_manifest(scene_number)
    if not workflow_path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = SceneWorkflowManifest.read(manifest_path)
    except (OSError, KeyError, TypeError, ValueError):
        return False
    return (
        manifest.pipeline == state.args.video_pipeline
        and not manifest.compare_canonical_dependencies(canonical_dependencies)
        and not manifest.verify(state.context.project_config_dir)
    )



def _select_render_scenes(state: PipelineRunState, scenes: tuple[RenderScene, ...]) -> tuple[RenderScene, ...]:
    selected = {state.args.smoke_scene} if state.args.smoke_only else parse_scene_list(state.args.scenes)
    return RenderPlan(scenes).select(scene_numbers=selected).scenes


def _missing_prepare_inputs(state: PipelineRunState, scenes: tuple[RenderScene, ...]) -> list[str]:
    missing: list[str] = []
    for path, label in ((state.plan_for_next_step, "render plan"), (state.context.input_audio, "audio")):
        if not path.is_file():
            missing.append(f"{label}: {path}")
    workflow = state.msr_workflow if state.args.video_pipeline == "ltx_msr" else state.ingredients_workflow
    if not workflow.is_file():
        missing.append(f"workflow template: {workflow}")
    for render_scene in scenes:
        scene = render_scene.to_dict()
        number = render_scene.scene_number
        for relay in (scene.get("ltx") or {}).get("prompt_relay") or []:
            if str(relay.get("state") or "").strip().lower() != "singing":
                continue
            prompt = str(relay.get("prompt") or "").lower()
            if "sing" not in prompt or ("lip sync" not in prompt and "lip-sync" not in prompt):
                missing.append(f"scene {number}: singing relay requires singing and lip sync")
        candidates: list[tuple[str, str]] = []
        if state.args.video_pipeline == "ltx_ingredients":
            ingredients = scene.get("ingredients") or {}
            sheet = ingredients.get("sheet_path") or scene.get("ingredients_scene_sheet")
            if sheet:
                candidates.append(("ingredients sheet", sheet))
            else:
                missing.append(f"scene {number}: ingredients_scene_sheet")
            anchors = ingredients.get("anchors") or scene.get("ingredients_scene_sheet_anchors") or []
            target = str(
                ingredients.get("global_prompt")
                or scene.get("ingredients_global_prompt")
                or scene.get("ingredients_target_prompt")
                or (scene.get("ltx") or {}).get("ingredients_target_prompt")
                or "",
            )
            references = scene.get("references") or {}
            expected_ids = {
                str(value) for value in references.get("actor_ids") or [] if str(value)
            }
            location_id = str(references.get("location_id") or "")
            if location_id:
                expected_ids.add(location_id)
            anchor_ids = {str(anchor.get("id") or "") for anchor in anchors}
            if expected_ids and expected_ids != anchor_ids:
                missing.append(f"scene {number}: ingredients anchors do not match actor/location bindings")
            unbound = sorted(item_id for item_id in anchor_ids if f"`{item_id}`" not in target)
            if unbound:
                missing.append(
                    f"scene {number}: global prompt does not bind anchors {', '.join(unbound)}",
                )
        else:
            references = scene.get("references") or {}
            actors = references.get("actor_msr_paths") or references.get("actor_sheet_paths") or []
            location = references.get("location_msr_path") or references.get("location_sheet_path")
            if not actors:
                missing.append(f"scene {number}: actor reference sheet")
            if not location:
                missing.append(f"scene {number}: location reference sheet")
            candidates.extend(("actor reference sheet", path) for path in actors)
            if location:
                candidates.append(("location reference sheet", location))
        for label, value in candidates:
            path = Path(value)
            path = path if path.is_absolute() else state.context.project_config_dir / path
            if not path.is_file():
                missing.append(f"scene {number} {label}: {path}")
    return missing


def _run_ltx_prepare_workflows_stage(state: PipelineRunState) -> None:
    if state.args.video_pipeline not in ("ltx_msr", "ltx_ingredients"):
        raise ValueError("prepare_workflows requires --video-pipeline ltx_msr or ltx_ingredients")
    all_scenes = _all_render_scenes(state) if state.plan_for_next_step.is_file() else []
    preflight = _run_visual_consistency_preflight(state, all_scenes)
    all_scenes = _project_visual_consistency_contracts(
        all_scenes,
        preflight,
    )
    scenes = _select_render_scenes(state, all_scenes)
    profile = _resolved_startframe_profile(state)
    handoff_predecessors = _music_handoff_predecessors(
        all_scenes,
        profile=profile,
    )
    if handoff_predecessors:
        selected_numbers = expand_handoff_selection(
            [
                contract
                for scene in all_scenes
                if (
                    contract := _stored_consistency_contract(
                        scene.to_dict(),
                    )
                )
                is not None
            ],
            {scene.scene_number for scene in scenes},
        )
        scenes = RenderPlan(all_scenes).select(
            scene_numbers=selected_numbers,
        ).scenes
    missing = _missing_prepare_inputs(state, scenes)
    if missing:
        raise FileNotFoundError("Cannot prepare scene workflows; missing inputs:\n- " + "\n- ".join(missing))
    backend = _specialized_video_use_case(state).backend
    materializer = WorkflowMaterializer(backend, state.context.artifact_layout)
    total = len(scenes)
    paths = [
        path
        for scene in scenes
        for path in (
            state.context.artifact_layout.scene_workflow(scene.scene_number),
            state.context.artifact_layout.scene_manifest(scene.scene_number),
        )
    ]
    previous = {path: path.read_bytes() if path.is_file() else None for path in paths}
    try:
        for completed, render_scene in enumerate(scenes, start=1):
            if render_scene.scene_number in handoff_predecessors:
                _report(
                    f"Deferred scene {completed}/{total}: "
                    f"{render_scene.scene_number} (awaiting predecessor handoff)",
                )
                continue
            canonical_dependencies = _canonical_dependencies_from_scene(
                render_scene.to_dict(),
            )
            if (
                canonical_dependencies is not None
                and _prepared_scene_is_fresh(
                    state,
                    render_scene.scene_number,
                    canonical_dependencies,
                )
            ):
                _report(
                    f"Reused prepared scene {completed}/{total}: "
                    f"{render_scene.scene_number}",
                )
                continue
            materializer.prepare(WorkflowMaterializationRequest(
                scene=render_scene.to_dict(),
                prompt=render_scene.video_prompt,
                audio_file=state.context.input_audio,
                render_plan_path=state.plan_for_next_step,
                pipeline=state.args.video_pipeline,
                canonical_dependencies=canonical_dependencies,
            ))
            _report(f"Prepared scene {completed}/{total}: {render_scene.scene_number}")
    except Exception:
        for path, content in previous.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)
        raise


def _attach_music_continuity_handoffs(
    state: PipelineRunState,
    *,
    all_scenes: list,
    selected_scenes: list,
    backend,
) -> list:
    if state.args.video_pipeline != "ltx_msr":
        return selected_scenes
    profile = _resolved_startframe_profile(state)
    if profile is None or not profile.supports_start_frame:
        return selected_scenes

    all_payloads = [scene.to_dict() for scene in all_scenes]
    by_number = {int(scene["scene"]): scene for scene in all_payloads}
    selected_numbers = {scene.scene_number for scene in selected_scenes}
    explicitly_selected = bool(
        {state.args.smoke_scene}
        if state.args.smoke_only
        else parse_scene_list(state.args.scenes),
    )
    attached = []
    for render_scene in selected_scenes:
        scene = render_scene.to_dict()
        number = render_scene.scene_number
        predecessor_id = str(scene.get("continuation_predecessor_id") or "").strip()
        previous = (
            next(
                (
                    payload for payload in all_payloads
                    if str(payload.get("technical_segment_id") or payload.get("segment_id") or "").strip()
                    == predecessor_id
                ),
                None,
            )
            if predecessor_id
            else by_number.get(number - 1)
        )
        previous_number = int(previous["scene"]) if previous is not None else number - 1
        previous_contract = _stored_consistency_contract(previous)
        current_contract = _stored_consistency_contract(scene)
        if (
            previous_contract is None
            or current_contract is None
            or (
                not predecessor_id
                and previous_contract.scene + 1 != current_contract.scene
            )
            or not can_handoff(previous_contract, current_contract)
        ):
            attached.append(render_scene)
            continue
        previous_clip = state.context.artifact_layout.scene_final_video(previous_number)
        if not previous_clip.is_file() and not explicitly_selected:
            attached.append(render_scene)
            continue
        output_frame = (
            state.context.render_dir
            / "keyframes"
            / f"scene_{previous_number:04}_to_{number:04}_start.png"
        )
        scene = ContinuityHandoffUseCase(
            PostprocessorFrameExtractor(
                backend.postprocessor,
                project_dir=state.context.project_config_dir,
                selected_rerender=explicitly_selected
                and number in selected_numbers,
            ),
        ).execute(
            previous_contract,
            current_contract,
            previous_clip,
            output_frame,
            scene,
            handoff_prompt=_music_handoff_prompt(previous),
        )
        attached.append(type(render_scene).from_dict(scene))
    return attached


def _stored_consistency_contract(scene: dict | None):
    payload = scene.get("visual_consistency") if scene else None
    if not isinstance(payload, dict):
        return None
    try:
        return SceneConsistencyContract.from_dict(payload)
    except (KeyError, TypeError, ValueError):
        return None


def _music_handoff_prompt(previous_scene: dict | None) -> str:
    ltx = (previous_scene or {}).get("ltx") or {}
    relays = ltx.get("msr_prompt_relay") or ltx.get("prompt_relay") or []
    if relays and isinstance(relays[-1], dict):
        prompt = str(relays[-1].get("prompt") or "").strip()
        if prompt:
            return prompt
    return str(
        ltx.get("original_style_i2v_prompt")
        or ltx.get("base_prompt")
        or "",
    ).strip()


_PROFILE_UNSET = object()


def _resolved_startframe_profile(
    state: PipelineRunState,
    profile_name: str | object | None = _PROFILE_UNSET,
):
    if state.args.video_pipeline != "ltx_msr":
        return None
    app_config = AppConfig.load(state.app_config_path)
    selected_name = (
        profile_name
        if profile_name is not _PROFILE_UNSET
        else getattr(state.args, "video_workflow_profile", None)
    )
    if profile_name is not _PROFILE_UNSET and selected_name is not None and not any(
        profile.name == selected_name
        for profile in app_config.video_workflow_profiles
    ):
        return None
    profile = app_config.resolve_video_workflow_profile(
        pipeline="ltx_msr",
        purpose="final",
        name=selected_name,
    )
    if profile is not None and profile.supports_start_frame:
        configured = Path(profile.workflow_path).resolve()
        materialized = Path(state.msr_workflow).resolve()
        if configured != materialized:
            raise ValueError(
                "Configured start-frame profile workflow does not match "
                f"the materialized MSR workflow: {configured} != {materialized}",
            )
    return profile


def _continuity_downstream(
    scene_number: int,
    predecessors: dict[int, int],
) -> set[int]:
    downstream: set[int] = set()
    current = scene_number
    while True:
        dependent = next(
            (
                candidate
                for candidate, predecessor in predecessors.items()
                if predecessor == current
            ),
            None,
        )
        if dependent is None:
            return downstream
        if dependent in downstream:  # cycle detection
            return downstream
        downstream.add(dependent)
        current = dependent


def _load_continuity_dirty(
    path: Path,
    scene_numbers: set[int],
) -> set[int]:
    if not path.exists() and not path.is_symlink():
        return set()
    if path.is_symlink() or not path.is_file():
        return set(scene_numbers)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        dirty = payload["dirty_scenes"]
        if (
            payload.get("schema") != "feverslop.continuity-dirty/v1"
            or not isinstance(dirty, list)
            or any(type(number) is not int for number in dirty)
            or not set(dirty).issubset(scene_numbers)
        ):
            raise ValueError("invalid continuity dirty marker")
        return set(dirty)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return set(scene_numbers)


def _write_continuity_dirty(
    path: Path,
    *,
    dirty_scenes: set[int],
    predecessor_scene: int,
    predecessor_contract: SceneConsistencyContract | None,
    predecessor_output: Path,
    project_dir: Path,
) -> None:
    project_root = Path(project_dir).resolve()
    marker = path.resolve()
    output = Path(predecessor_output).resolve()
    if (
        not marker.is_relative_to(project_root)
        or not output.is_relative_to(project_root)
    ):
        raise ValueError("Continuity dirty state must remain inside project")
    relative_output = output.relative_to(project_root).as_posix()
    payload = {
        "schema": "feverslop.continuity-dirty/v1",
        "dirty_scenes": sorted(dirty_scenes),
        "predecessor": {
            "scene": predecessor_scene,
            "fingerprint": (
                predecessor_contract.fingerprint
                if predecessor_contract is not None
                else None
            ),
            "output": {
                "path": relative_output,
                "sha256": (
                    hashlib.sha256(output.read_bytes()).hexdigest()
                    if output.is_file()
                    else None
                ),
            },
        },
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=marker.parent,
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, marker)
    finally:
        temporary.unlink(missing_ok=True)


def _run_visual_consistency_preflight(
    state: PipelineRunState,
    scenes: list,
) -> VisualConsistencyPreflightResult:
    preflight_mode = getattr(
        state.args,
        "visual_consistency_preflight",
        PreflightMode.WARN,
    )
    preflight_mode = PreflightMode.parse(preflight_mode)
    if preflight_mode is PreflightMode.OFF:
        return VisualConsistencyPreflightResult((), ())
    project_config = ProjectConfig.load(state.context.project_config_path)
    snapshot = ProjectReferenceManifestAdapter(
        lambda _project_id: state.context.project_config_dir,
    ).load(state.context.project_config_dir.name)
    mode = "msr" if state.args.video_pipeline == "ltx_msr" else "ingredients"
    workflow = (
        state.msr_workflow
        if state.args.video_pipeline == "ltx_msr"
        else state.ingredients_workflow
    )
    scene_payloads = [scene.to_dict() for scene in scenes]
    workflow_profile = resolve_preflight_workflow_profile(
        scene_payloads,
        explicit_profile=getattr(state.args, "video_workflow_profile", None),
        legacy_fallback=workflow.stem,
    )
    selected_profile = (
        _resolved_startframe_profile(state, workflow_profile)
        if mode == "msr"
        else None
    )
    result = preflight_visual_consistency(
        scene_payloads,
        snapshot,
        mode=mode,
        workflow_profile=workflow_profile,
        preflight_mode=preflight_mode,
        subject_mode=project_config.subject_mode,
        max_scene_actors=project_config.max_scene_actors,
        supports_continuous_transitions=(
            mode == "msr"
            and selected_profile is not None
            and selected_profile.supports_start_frame
        ),
    )
    artifact_issues = validate_project_scene_artifacts(
        state.context.project_config_dir,
        scene_payloads,
        mode=mode,
        preflight_mode=preflight_mode,
    )
    result = VisualConsistencyPreflightResult(
        result.contracts,
        (*result.issues, *artifact_issues),
    )
    for issue in result.issues:
        _report(
            f"Visual consistency {issue.severity.upper()} "
            f"scene {issue.scene} {issue.code}: {issue.message}",
        )
    if not result.renderable:
        details = "\n- ".join(
            f"{issue.code}: {issue.message}"
            for issue in result.issues
            if issue.severity == "error"
        )
        raise ValueError(
            "Visual consistency preflight blocked workflow preparation:\n- "
            + details,
        )
    return result


def _project_visual_consistency_contracts(
    scenes: list,
    result: VisualConsistencyPreflightResult,
) -> list:
    """Attach canonical preflight contracts to in-memory render scenes only."""
    if not isinstance(result, VisualConsistencyPreflightResult):
        return list(scenes)
    contracts = {contract.scene: contract for contract in result.contracts}
    projected = []
    for scene in scenes:
        payload = scene.to_dict()
        contract = contracts.get(scene.scene_number)
        if contract is not None:
            payload["visual_consistency"] = contract.to_dict()
        projected.append(type(scene).from_dict(payload))
    return projected


def _music_handoff_predecessors(
    scenes: list,
    *,
    profile,
) -> dict[int, int]:
    if profile is None or not profile.supports_start_frame:
        return {}
    predecessors: dict[int, int] = {}
    payloads = [scene.to_dict() for scene in scenes]
    for previous_scene, current_scene in zip(payloads, payloads[1:]):
        previous_contract = _stored_consistency_contract(previous_scene)
        current_contract = _stored_consistency_contract(current_scene)
        if (
            previous_contract is not None
            and current_contract is not None
            and previous_contract.scene + 1 == current_contract.scene
            and can_handoff(previous_contract, current_contract)
        ):
            predecessors[current_contract.scene] = previous_contract.scene
    technical_by_id = {
        str(payload.get("technical_segment_id") or payload.get("segment_id") or "").strip(): payload
        for payload in payloads
    }
    for current_scene in payloads:
        predecessor_id = str(current_scene.get("continuation_predecessor_id") or "").strip()
        if not predecessor_id:
            continue
        previous_scene = technical_by_id.get(predecessor_id)
        if previous_scene is not None:
            predecessors[int(current_scene["scene"])] = int(previous_scene["scene"])
    return predecessors


def _continuation_boundary_manifest_valid(
    state: PipelineRunState,
    scene: RenderScene,
    handoff_predecessors: dict[int, int],
) -> bool:
    """Verify the persisted anchor evidence before releasing a successor."""
    predecessor = handoff_predecessors.get(scene.scene_number)
    if predecessor is None:
        return True
    keyframes = scene.to_dict().get("keyframes") or {}
    manifest = keyframes.get("boundary_frame_manifest")
    if not isinstance(manifest, dict):
        # Older prepared manifests do not carry the handoff payload. Keep
        # resume compatible when both sides of the boundary are present; new
        # handoffs always take the manifest-backed path below.
        return (
            state.context.artifact_layout.scene_final_video(predecessor).is_file()
            and state.context.artifact_layout.scene_final_video(scene.scene_number).is_file()
        )
    try:
        frame_path = Path(manifest["frame_path"])
        if not frame_path.is_absolute():
            frame_path = state.context.project_config_dir / frame_path
        return frame_path.is_file()
    except (KeyError, TypeError, ValueError, OSError):
        return False


def _run_ltx_render_scenes_stage(state: PipelineRunState) -> None:
    if state.args.video_pipeline == "minimax-h3-r2v":
        # An explicit --stage list may place rendering before
        # msr_reference_sheets. Refresh the plan from the existing enriched
        # plan immediately before scene validation so actor paths cannot be
        # lost merely because the stages were listed in that order.
        _preserve_enriched_reference_paths(
            output_path=state.plan_for_next_step,
            reference_plan_path=state.context.reference_plan,
        )
    if state.args.video_pipeline not in ("ltx_msr", "ltx_ingredients") and state.args.render_mode != "single_prompt" and not str(state.args.relay_workflow).strip():
        raise ValueError(f"RenderMode '{state.args.render_mode}' requires --relay-workflow pointing to a workflow with #PROMPT_RELAY.")

    if state.args.video_pipeline in ("ltx_msr", "ltx_ingredients"):
        all_scenes = _all_render_scenes(state)
        preflight = _run_visual_consistency_preflight(state, all_scenes)
        all_scenes = _project_visual_consistency_contracts(
            all_scenes,
            preflight,
        )
        scenes = _select_render_scenes(state, all_scenes)
        profile = _resolved_startframe_profile(state)
        handoff_predecessors = _music_handoff_predecessors(
            all_scenes,
            profile=profile,
        )
        if profile is not None and profile.supports_start_frame:
            contracts = [
                contract
                for scene in all_scenes
                if (
                    contract := _stored_consistency_contract(
                        scene.to_dict(),
                    )
                )
                is not None
            ]
            selected_numbers = expand_handoff_selection(
                contracts,
                {scene.scene_number for scene in scenes},
            )
            scenes = RenderPlan(all_scenes).select(
                scene_numbers=selected_numbers,
            ).scenes
        missing: list[Path] = []
        for scene in scenes:
            workflow_path = state.context.artifact_layout.scene_workflow(scene.scene_number)
            manifest_path = state.context.artifact_layout.scene_manifest(scene.scene_number)
            deferred = scene.scene_number in handoff_predecessors
            if not workflow_path.is_file() and (
                not deferred or manifest_path.is_file()
            ):
                missing.append(workflow_path)
            if not manifest_path.is_file() and (
                not deferred or workflow_path.is_file()
            ):
                missing.append(manifest_path)
        if missing:
            raise FileNotFoundError(
                "Missing prepared scene workflows: " + ", ".join(str(path) for path in missing)
                + ". Run --stage ltx_prepare_workflows first.",
            )
        backend = _specialized_video_use_case(state).backend
        active_workflow_profile = (
            profile.name
            if profile is not None
            else resolve_preflight_workflow_profile(
                [scene.to_dict() for scene in all_scenes],
                explicit_profile=getattr(
                    state.args,
                    "video_workflow_profile",
                    None,
                ),
                legacy_fallback=(
                    state.msr_workflow
                    if state.args.video_pipeline == "ltx_msr"
                    else state.ingredients_workflow
                ).stem,
            )
        )
        renderer = PreparedWorkflowRenderer(
            project_dir=state.context.project_config_dir,
            render_queue=backend.render_queue,
            postprocessor=backend.postprocessor,
            expected_pipeline=state.args.video_pipeline,
            expected_workflow_profile=active_workflow_profile,
            max_render_frames=backend.max_render_frames,
            max_render_duration_seconds=backend.max_render_duration_seconds,
            render_budget_workflow_path=backend.render_budget_workflow_path,
            round_render_frames_to_8n1=backend.round_render_frames_to_8n1,
            asset_uploader=backend.asset_uploader,
            model_resolver=backend.model_resolver,
            model_workflow_path=backend.workflow_label,
        )
        total = len(scenes)
        materializer = WorkflowMaterializer(
            backend,
            state.context.artifact_layout,
        )
        explicit_selection = bool(
            {state.args.smoke_scene}
            if state.args.smoke_only
            else parse_scene_list(state.args.scenes),
        )
        rendered_this_run: set[int] = set()
        dirty_marker = (
            state.context.render_dir / "continuity_dirty.json"
        )
        persisted_dirty = _load_continuity_dirty(
            dirty_marker,
            {scene.scene_number for scene in all_scenes},
        )
        with RenderProgressReporter(
            "Rendering prepared LTX scenes", total, emit_scene_progress=True,
        ) as progress:
            completed = 0
            rendered_scene: RenderScene | None = None

            def render_one(scene: RenderScene) -> bool:
                nonlocal completed, rendered_scene
                completed += 1
                rendered_scene = scene
                workflow = state.context.artifact_layout.scene_workflow(scene.scene_number)
                final_path = state.context.artifact_layout.scene_final_video(scene.scene_number)
                predecessor_rendered = (
                    handoff_predecessors.get(scene.scene_number)
                    in rendered_this_run
                )
                skip_existing = (
                    False
                    if (
                        state.args.smoke_only
                        or explicit_selection
                        or predecessor_rendered
                        or scene.scene_number in persisted_dirty
                        or not workflow.is_file()
                    )
                    else not state.args.no_skip_existing
                )
                if not (skip_existing and file_is_valid(final_path)):
                    randomize_seed = bool(getattr(backend, "randomize_seed", False))
                    if randomize_seed:
                        scene_payload = scene.to_dict()
                        new_seed = random.SystemRandom().randint(0, 2**63 - 1)
                        scene_payload["seed"] = new_seed
                        plan_data = JsonArtifactStore().read_render_plan(state.plan_for_next_step)
                        plan_data = patch_render_plan_seed(
                            plan_data,
                            scene_number=scene.scene_number,
                            seed=new_seed,
                        )
                        JsonArtifactStore().write_render_plan(
                            state.plan_for_next_step,
                            plan_data,
                        )
                        scene = RenderPlan.from_dicts([scene_payload]).scenes[0]
                    downstream = _continuity_downstream(
                        scene.scene_number,
                        handoff_predecessors,
                    )
                    if downstream:
                        persisted_dirty.update(downstream)
                        _write_continuity_dirty(
                            dirty_marker,
                            dirty_scenes=persisted_dirty,
                            predecessor_scene=scene.scene_number,
                            predecessor_contract=(
                                _stored_consistency_contract(
                                    scene.to_dict(),
                                )
                            ),
                            predecessor_output=final_path,
                            project_dir=state.context.project_config_dir,
                        )
                    [render_scene] = _attach_music_continuity_handoffs(
                        state,
                        all_scenes=all_scenes,
                        selected_scenes=[scene],
                        backend=backend,
                    )
                    rendered_scene = render_scene
                    original_randomize_seed = backend.randomize_seed
                    if randomize_seed:
                        backend.randomize_seed = False
                    try:
                        if randomize_seed or render_scene.to_dict() != scene.to_dict():
                            materializer.prepare(
                                WorkflowMaterializationRequest(
                                    scene=render_scene.to_dict(),
                                    prompt=render_scene.video_prompt,
                                    audio_file=state.context.input_audio,
                                    render_plan_path=state.plan_for_next_step,
                                    pipeline=state.args.video_pipeline,
                                    canonical_dependencies=_canonical_dependencies_from_scene(
                                        render_scene.to_dict(),
                                    ),
                                ),
                            )
                        canonical_dependencies = _canonical_dependencies_from_scene(
                            render_scene.to_dict(),
                        )
                        final_path = (
                            renderer.render(workflow)
                            if canonical_dependencies is None
                            else renderer.render(
                                workflow,
                                canonical_dependencies=canonical_dependencies,
                            )
                        )
                    finally:
                        backend.randomize_seed = original_randomize_seed
                    rendered_this_run.add(scene.scene_number)
                    if downstream:
                        _write_continuity_dirty(
                            dirty_marker,
                            dirty_scenes=persisted_dirty,
                            predecessor_scene=scene.scene_number,
                            predecessor_contract=(
                                _stored_consistency_contract(
                                    scene.to_dict(),
                                )
                            ),
                            predecessor_output=final_path,
                            project_dir=state.context.project_config_dir,
                        )
                    if scene.scene_number in persisted_dirty:
                        persisted_dirty.remove(scene.scene_number)
                        if persisted_dirty:
                            _write_continuity_dirty(
                                dirty_marker,
                                dirty_scenes=persisted_dirty,
                                predecessor_scene=scene.scene_number,
                                predecessor_contract=(
                                    _stored_consistency_contract(
                                        scene.to_dict(),
                                    )
                                ),
                                predecessor_output=final_path,
                                project_dir=state.context.project_config_dir,
                            )
                        else:
                            dirty_marker.unlink(missing_ok=True)
                else:
                    canonical_dependencies = _canonical_dependencies_from_scene(
                        scene.to_dict(),
                    )
                    if canonical_dependencies is not None:
                        renderer.verify_canonical_dependencies(
                            workflow,
                            canonical_dependencies,
                        )
                    manifest = SceneWorkflowManifest.read(state.context.artifact_layout.scene_manifest(scene.scene_number))
                    if manifest.pipeline != state.args.video_pipeline:
                        raise ValueError(
                            f"Prepared workflow pipeline {manifest.pipeline!r} does not match "
                            f"expected pipeline {state.args.video_pipeline!r}",
                        )
                    mismatches = manifest.verify(state.context.project_config_dir)
                    if mismatches:
                        raise ValueError("Prepared workflow verification failed: " + "; ".join(mismatches))

                progress.update(final_path, completed, total)
                return _continuation_boundary_manifest_valid(
                    state,
                    rendered_scene or scene,
                    handoff_predecessors,
                )

            scene_by_id = {f"scene-{scene.scene_number}": scene for scene in scenes}
            chains = chains_from_predecessors(
                [scene.scene_number for scene in scenes],
                handoff_predecessors,
            )
            scheduler = ContinuationScheduler(
                chains,
                reporter=ConsoleReporter(console),
            )
            scheduler.run(
                lambda segment_id: render_one(scene_by_id[segment_id]),
            )
        return

    selected_workflows = _selected_video_workflows(state)
    if state.args.video_pipeline in ("ltx_msr", "ltx_ingredients"):
        ltx_workflow = selected_workflows[0]
        ltx_single_prompt_workflow = None
    else:
        ltx_workflow = selected_workflows[0]
        ltx_single_prompt_workflow = selected_workflows[1] if state.args.render_mode == "auto" else None
    video_use_case = build_render_video_scenes_use_case(
        RenderVideoCompositionOptions(
            app_config_path=state.app_config_path,
            project_config_path=state.context.project_config_path,
            render_plan_path=state.plan_for_next_step,
            workflow_path=ltx_workflow,
            output_dir=state.context.ltx_dir,
            video_pipeline=state.args.video_pipeline,
            single_prompt_workflow_path=ltx_single_prompt_workflow,
            render_mode=state.args.render_mode,
            single_prompt_title=state.args.single_prompt_title,
            single_prompt_input=state.args.single_prompt_input,
            character_lora_strength=state.args.video_character_lora_strength,
            lora_1_strength_model=state.args.video_lora_1_strength_model,
            lora_1_strength_clip=state.args.video_lora_1_strength_clip,
            lora_split_enabled=state.args.lora_split_enabled,
            randomize_seed=state.args.randomize_seed,
            debug_workflows_dir=state.context.ltx_debug_dir,
            rolling_frame_profile=state.args.rolling_frame_profile,
            resolution=_get_resolution(state.args),
        ),
        console=console,
    )
    ltx_scene_numbers = {state.args.smoke_scene} if state.args.smoke_only else parse_scene_list(state.args.scenes)
    ltx_total = count_render_plan_items(state.plan_for_next_step, scene_numbers=ltx_scene_numbers)
    with RenderProgressReporter(
        VIDEO_SCENE_PROGRESS_LABEL, ltx_total, emit_scene_progress=True,
    ) as ltx_progress:
        video_use_case.execute(
            RenderVideoScenesRequest(
                render_plan_path=state.plan_for_next_step,
                canonical_plan_path=_canonical_plan_path(state),
                workflow_path=ltx_workflow,
                audio_file=state.context.input_audio,
                storyboard_dir=state.context.storyboard_dir,
                output_dir=state.context.ltx_dir,
                render_mode=state.args.render_mode,
                single_prompt_workflow_path=ltx_single_prompt_workflow,
                scene_numbers=ltx_scene_numbers,
                skip_existing=False if state.args.smoke_only else not state.args.no_skip_existing,
                anchors=WorkflowAnchorConfig(
                    single_prompt_title=state.args.single_prompt_title,
                    single_prompt_input=state.args.single_prompt_input,
                ),
                on_scene_complete=ltx_progress.update,
            ),
        )


def _assemble_declared_cutless_groups(
    render_plan: list[dict],
    clips: list[Path],
    *,
    output_dir: Path,
    postprocessor: VideoPostProcessor,
) -> list[Path]:
    """Replace explicitly declared continuation groups with hard-cut movies."""
    clips_by_segment = {
        str((entry.get("metadata") or {}).get("segment_id") or entry.get("segment_id") or ""): clip
        for entry, clip in zip(render_plan, clips)
    }
    groups: list[dict] = []
    seen: set[str] = set()
    for entry in render_plan:
        for group in (entry.get("metadata") or {}).get("continuation_groups") or []:
            group_id = str(group.get("group_id") or "").strip()
            if group_id and group_id not in seen:
                groups.append(group)
                seen.add(group_id)
    if not groups:
        return clips

    assembled = list(clips)
    for index, group in enumerate(groups, start=1):
        segment_ids = [
            str(segment.get("segment_id") or "").strip()
            for segment in group.get("segments") or []
        ]
        if not segment_ids or any(segment_id not in clips_by_segment for segment_id in segment_ids):
            _report(
                f"[yellow]Skipping cutless group {group.get('group_id', index)}: "
                "rendered segment clips are not individually addressable.[/yellow]",
            )
            continue
        group_clips = {segment_id: clips_by_segment[segment_id] for segment_id in segment_ids}
        output_file = output_dir / f"cutless_{index:04d}.mp4"
        diagnostics_file = output_dir / f"cutless_{index:04d}.json"
        assemble_declared_cutless_group(
            group=group,
            clips_by_segment=group_clips,
            frame_count=postprocessor._frame_count,
            extract_first_frame=postprocessor.extract_first_frame,
            extract_last_frame=postprocessor.extract_last_frame,
            assembly_service=CutlessAssemblyService(postprocessor),
            output_file=output_file,
            diagnostics_file=diagnostics_file,
            fps=int(render_plan[0].get("fps") or 24),
            duplicate_policy=str(group.get("duplicate_policy") or "reject"),
            reporter=ConsoleReporter(console),
        )
        first_clip_index = min(assembled.index(clips_by_segment[segment_id]) for segment_id in segment_ids)
        assembled = [
            clip for clip in assembled
            if clip not in group_clips.values() or clip == assembled[first_clip_index]
        ]
        assembled[first_clip_index] = output_file
    return assembled


def _stage_final_postprocessor(state: PipelineRunState) -> VideoPostProcessor:
    """Build the final-assembly postprocessor with the project FFmpeg timeout."""
    return final_video_postprocessor(
        AppConfig.load(state.app_config_path).comfyui.ffmpeg_timeout_seconds
    )


def _run_concat_video_only_stage(state: PipelineRunState) -> None:
    from feverslop.domain.scene_recovery import require_ready_scenes
    from .config_loader import (
        collect_render_plan_scene_clips,
        collect_render_plan_scene_raw_clips,
        rewrite_concat_list,
        write_concat_list,
    )
    layout = state.context.artifact_layout
    render_plan = json.loads(Path(state.plan_for_next_step).read_text(encoding="utf-8-sig"))
    args = getattr(state, "args", None)
    selected_scenes = parse_scene_list(getattr(args, "scenes", None))
    if selected_scenes is None and getattr(args, "smoke_only", False):
        selected_scenes = {int(args.smoke_scene)}
    render_plan = RenderPlan.from_dicts(render_plan).select(
        scene_numbers=selected_scenes,
    ).to_dicts()
    require_ready_scenes(
        render_plan,
        project_path=getattr(getattr(state, "context", None), "project_config_dir", None),
    )
    scene_numbers = [int(entry["scene"]) for entry in render_plan]
    canonical_clips = [layout.scene_final_video(scene_number) for scene_number in scene_numbers]
    canonical_available = [clip for clip in canonical_clips if clip.is_file()]
    if canonical_available and len(canonical_available) != len(canonical_clips):
        missing = [clip.parent.name for clip in canonical_clips if not clip.is_file()]
        raise FileNotFoundError(
            "Cannot build base variant without mixing artifact layouts; missing canonical "
            f"final.mp4 for {', '.join(missing[:10])}",
        )
    if canonical_available:
        clips = canonical_clips
    else:
        _report(
            "[yellow]No canonical final.mp4 scene artifacts found; using the complete legacy "
            "scene layout for the base movie.[/yellow]",
        )
        clips = collect_render_plan_scene_clips(
            state.plan_for_next_step,
            state.context.ltx_dir,
            layout=None,
        )
    clips = _assemble_declared_cutless_groups(
        render_plan,
        clips,
        output_dir=layout.final_dir,
        postprocessor=_stage_final_postprocessor(state),
    )
    rewrite_concat_list(clips, state.context.artifact_layout.final_dir)
    postprocessor = _stage_final_postprocessor(state)
    _report(f"Concatenating base variant: {len(clips)} scene clips")
    state.video_only_path = postprocessor.concat_clips(
        concat_list=state.context.concat_list,
        output_file=layout.video_only,
        video_only=True,
    )
    state.video_only_variants = {"base": state.video_only_path}

    optional_variants = (
        ("facefix", layout.scene_final_facefix_video, layout.video_only_facefix),
        ("upscaled", layout.scene_upscaled_video, layout.video_only_upscaled),
    )
    for variant, scene_path, output_path in optional_variants:
        variant_clips = [scene_path(scene_number) for scene_number in scene_numbers]
        available = [clip for clip in variant_clips if clip.is_file()]
        if not available:
            continue
        if len(available) != len(variant_clips):
            missing = [clip.parent.name for clip in variant_clips if not clip.is_file()]
            _report(
                f"[yellow]Skipping {variant} movie: found {len(available)}/{len(variant_clips)} "
                f"scene clips; missing {', '.join(missing[:10])}.[/yellow]",
            )
            continue
        concat_list = write_concat_list(
            variant_clips,
            layout.final_dir,
            f"concat_{variant}.txt",
        )
        _report(f"Concatenating {variant} variant: {len(variant_clips)} scene clips")
        state.video_only_variants[variant] = postprocessor.concat_clips(
            concat_list=concat_list,
            output_file=output_path,
            video_only=True,
            reencode=variant == "upscaled",
        )

    try:
        raw_clips = collect_render_plan_scene_raw_clips(
            state.plan_for_next_step,
            state.context.ltx_dir,
            layout=state.context.artifact_layout,
        )
    except FileNotFoundError:
        raw_clips = clips
    if raw_clips is not clips:
        write_concat_list(raw_clips, state.context.artifact_layout.final_dir, "concat_raw.txt")
    else:
        state.context.concat_raw.write_text(
            state.context.concat_list.read_text(encoding="utf-8"), encoding="utf-8",
        )
    postprocessor.concat_clips(
        concat_list=state.context.concat_raw,
        output_file=layout.video_audio,
        video_only=False,
    )


def _run_mux_original_audio_stage(state: PipelineRunState) -> None:
    from feverslop.domain.scene_recovery import require_ready_scenes
    if getattr(state, "plan_for_next_step", None):
        require_ready_scenes(
            json.loads(Path(state.plan_for_next_step).read_text(encoding="utf-8-sig")),
            project_path=getattr(getattr(state, "context", None), "project_config_dir", None),
        )
    layout = getattr(state.context, "artifact_layout", None)
    variants = getattr(state, "video_only_variants", None)
    if layout is not None and not variants:
        variants = {}
        if layout.video_only.is_file():
            variants["base"] = layout.video_only
        render_plan = json.loads(Path(state.plan_for_next_step).read_text(encoding="utf-8-sig"))
        scene_numbers = [int(entry["scene"]) for entry in render_plan]
        optional_variants = (
            ("facefix", layout.scene_final_facefix_video, layout.video_only_facefix),
            ("upscaled", layout.scene_upscaled_video, layout.video_only_upscaled),
        )
        for variant, scene_path, aggregate_path in optional_variants:
            if not aggregate_path.is_file():
                continue
            scene_clips = [scene_path(scene_number) for scene_number in scene_numbers]
            if all(clip.is_file() for clip in scene_clips):
                variants[variant] = aggregate_path
            else:
                _report(
                    f"[yellow]Ignoring stale {aggregate_path.name}: the current render plan "
                    f"does not have a complete {variant} scene set.[/yellow]",
                )
    if variants:
        output_paths = {
            "base": layout.movie,
            "facefix": layout.movie_facefix,
            "upscaled": layout.movie_upscaled,
        }
        postprocessor = _stage_final_postprocessor(state)
        results: dict[str, Path] = {}
        for variant in ("base", "facefix", "upscaled"):
            video_file = variants.get(variant)
            if video_file is None:
                continue
            _report(f"Muxing original audio into {variant} movie")
            results[variant] = postprocessor.mux_original_audio(
                video_file=video_file,
                audio_file=state.context.input_audio,
                output_file=output_paths[variant],
            )
        state.final_video_path = results.get("base") or next(iter(results.values()))
        return

    video_only_path = state.video_only_path or state.context.final_concat_video
    if state.video_only_path is None and not Path(video_only_path).exists():
        raise FileNotFoundError(f"Video-only concat not found: {video_only_path}")
    postprocessor = _stage_final_postprocessor(state)
    output_file = state.context.final_concat
    if Path(video_only_path).name == "video_only_upscaled.mp4":
        output_file = Path(state.context.final_concat).with_name("movie_upscaled.mp4")
    state.final_video_path = postprocessor.mux_original_audio(
        video_file=video_only_path,
        audio_file=state.context.input_audio,
        output_file=output_file,
    )


def _run_upscale_stage(state: PipelineRunState) -> None:
    from feverslop.adapters.comfyui_seedvr2_backend import ComfyUISeedVR2Backend

    from .seedvr2_pipeline import SeedVR2CompositionOptions, run_seedvr2

    config = ProjectConfig.load(state.context.project_config_path)
    if not config.upscale.enabled and not getattr(state.args, "upscale", False):
        _report("SeedVR2 upscale disabled in project config.")
        return
    config.upscale.validate_resources()
    workflow_path = Path(config.upscale.workflow_path)
    if not workflow_path.is_absolute():
        workflow_path = runner_root() / workflow_path
    backend = ComfyUISeedVR2Backend(
        client=state.comfyui_client,
        workflow_path=workflow_path,
    )
    reporter = ConsoleReporter(console)
    run_seedvr2(SeedVR2CompositionOptions(
        project_config_path=state.context.project_config_path,
        render_plan_path=state.plan_for_next_step,
        backend=backend,
        skip_existing=not state.args.no_skip_existing,
        force_enabled=bool(getattr(state.args, "upscale", False)),
        resolution_override=(
            (state.args.upscale_resolution.width, state.args.upscale_resolution.height)
            if getattr(state.args, "upscale_resolution", None) is not None
            else None
        ),
        scene_numbers=parse_scene_list(getattr(state.args, "scenes", None)),
        reporter=reporter,
    ))
    _report("[green]SeedVR2 upscale artifacts ready.[/green]")


def _run_diagnostic_scene_audio_concat_stage(state: PipelineRunState) -> None:
    postprocessor = _stage_final_postprocessor(state)
    postprocessor.concat_clips(
        concat_list=state.context.concat_list,
        output_file=state.context.final_concat_scene_audio_debug,
        video_only=False,
    )


def _run_timeline_export_stage(state: PipelineRunState) -> None:
    from .config_loader import collect_render_plan_scene_clips

    config = ProjectConfig.load(state.context.project_config_path)
    video = config.to_video_settings()
    legacy_openshot_stage = PipelineStage.OPENSHOT_EXPORT.value in (getattr(state.args, "stages", None) or [])
    requested_format = "openshot" if legacy_openshot_stage else getattr(state.args, "timeline_format", "both")
    export_formats = [requested_format] if requested_format != "both" else ["mlt", "openshot"]
    layout = state.context.artifact_layout
    plan_entries = json.loads(Path(state.plan_for_next_step).read_text(encoding="utf-8-sig"))
    has_facefix = any(
        layout.scene_final_facefix_video(int(entry.get("scene") or entry.get("scene_number"))).is_file()
        for entry in plan_entries
    )
    has_upscaled = any(
        layout.scene_upscaled_video(int(entry.get("scene") or entry.get("scene_number"))).is_file()
        for entry in plan_entries
    )
    variants = [("", False, False)]
    if has_facefix:
        variants.append(("_facefix", True, False))
    if has_upscaled:
        variants.append(("_upscaled", False, True))

    def report(completed: int, total: int, label: str) -> None:
        _report(f"[dim]Timeline export: {completed}/{total} ({label})[/dim]")

    for export_format in export_formats:
        extension = "mlt" if export_format == "mlt" else "osp"
        output_dir_name = "timeline" if export_format == "mlt" else "openshot"
        for suffix, prefer_facefix, prefer_upscaled in variants:
            clips = collect_render_plan_scene_clips(
                state.plan_for_next_step,
                state.context.ltx_dir,
                layout=layout,
                prefer_facefix=prefer_facefix,
                prefer_upscaled=prefer_upscaled,
            )
            output_path = state.context.project_output_dir / output_dir_name / f"{state.context.project_file_stem}{suffix}.{extension}"
            _report(f"Timeline export ({export_format}, {suffix or 'final'}): writing {len(clips)} rendered clips")
            if export_format == "mlt":
                written = export_render_plan_to_mlt(
                    render_plan_path=state.plan_for_next_step, clip_paths=clips,
                    audio_path=state.context.input_audio, output_path=output_path,
                    width=video.width, height=video.height, fps=video.fps,
                    project_name=f"{state.context.project_file_stem}{suffix}",
                )
                if not suffix:
                    state.timeline_project_path = written
            else:
                written = export_render_plan_to_openshot(
                    render_plan_path=state.plan_for_next_step, clip_paths=clips,
                    audio_path=state.context.input_audio, output_path=output_path,
                    width=video.width, height=video.height, fps=video.fps,
                    on_progress=report,
                )
                if not suffix:
                    state.openshot_project_path = written
            _report(f"[green]Timeline project written: {output_path}[/green]")


def _run_facefix_stage(state: PipelineRunState) -> None:
    from feverslop.composition.facefix_pipeline import (
        FaceFixCompositionOptions,
        run_facefix,
    )

    layout = state.context.artifact_layout
    scenes_dir = layout.scenes_dir
    if not scenes_dir.is_dir():
        _report(
            "[yellow]No scenes directory found at"
             f" {scenes_dir}, FaceFix skipped.[/yellow]",
        )
        return

    scene_numbers = None
    if state.args.scenes:
        scene_numbers = sorted(parse_scene_list(state.args.scenes))

    options = FaceFixCompositionOptions(
        app_config_path=str(state.app_config_path),
        workflow_path=str(state.facefix_workflow),
        scenes_dir=str(scenes_dir),
        project_dir=str(state.context.project_config_dir),
        scene_numbers=scene_numbers,
        reference_images=layout.actor_sheet_images(),
        skip_existing=not state.args.no_skip_existing,
        ffmpeg_debug=getattr(state.args, "facefix_debug", False),
        use_crop_pipeline=True,
    )
    run_facefix(options, console=console)


def _run_facefix_concat_stage(state: PipelineRunState) -> None:
    _report("FaceFix final concat uses the shared artifact-variant assembler.")
    _run_concat_video_only_stage(state)
    _run_mux_original_audio_stage(state)


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
