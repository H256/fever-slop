"""Render-stage runners (LTX prepare, render scenes, continuity) for the
M-20 split of ``stage_runners`` (issue #1194).

Moved mechanically from ``feverslop.composition.stage_runners`` (P4).
``stage_runners`` re-exports these so existing imports and
``patch("...stage_runners.X")`` targets keep working.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from tempfile import NamedTemporaryFile

from feverslop.adapters.local_artifacts import JsonArtifactStore
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
from feverslop.application.continuity_handoff import ContinuityHandoffUseCase
from feverslop.application.effective_render_plan import (
    CanonicalSceneDependencies,
    project_effective_plan,
)
from feverslop.application.render_video import (
    RenderVideoScenesRequest,
    patch_render_plan_seed,
)
from feverslop.application.visual_consistency_preflight import (
    VisualConsistencyPreflightResult,
    preflight_visual_consistency,
    resolve_preflight_workflow_profile,
)
from feverslop.composition.continuation_scheduler import (
    ContinuationScheduler,
    chains_from_predecessors,
)
from feverslop.composition.render_video import (
    RenderVideoCompositionOptions,
    build_render_video_scenes_use_case,
)
from feverslop.config.app_config import AppConfig
from feverslop.config.project_config import ProjectConfig
from feverslop.domain.prepared_workflow import SceneWorkflowManifest
from feverslop.domain.render_plan import RenderPlan, RenderScene
from feverslop.domain.semantic_intent import persisted_ledger_for_project
from feverslop.domain.visual_consistency import (
    PreflightMode,
    SceneConsistencyContract,
    can_handoff,
    expand_handoff_selection,
    validate_scene_sequence,
)
from feverslop.ports.rendering import WorkflowAnchorConfig
from feverslop.tools.storyboard_page import (
    parse_scene_list,
)
from feverslop.utils.io import file_is_valid

from ..config_loader import PipelineRunState, count_render_plan_items
from .plan_stages import (
    _get_resolution,
    _preserve_enriched_reference_paths,
    _selected_video_workflows,
)
from .progress import (
    RenderProgressReporter,
    VIDEO_SCENE_PROGRESS_LABEL,
    _canonical_plan_path,
    _report,
    console,
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


def _all_render_scenes(state: PipelineRunState) -> tuple[RenderScene, ...]:
    payload = JsonArtifactStore().read_json(state.plan_for_next_step)
    canonical_path = _canonical_plan_path(state)
    canonical_payload = (
        JsonArtifactStore().read_json(canonical_path)
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
    app_config.attach_import_store(state.context.project_config_dir)
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
        ensembles=project_config.ensembles,
        semantic_intent_ledger=persisted_ledger_for_project(
            project_config.project_dir,
            project_config.song_id,
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
