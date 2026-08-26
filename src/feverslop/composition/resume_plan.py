from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from feverslop.adapters.canonical_plan_store import CanonicalPlanStore
from feverslop.application.canonical_plan_migration import analyze_canonical_plan_migration
from feverslop.application.effective_render_plan import project_effective_plan
from feverslop.application.msr_prompt_enrichment import msr_prompt_input_fingerprint
from feverslop.domain.canonical_render_plan import PromptRole
from feverslop.domain.effective_render_plan import CanonicalSceneDependencies
from feverslop.domain.execution_plan import ExecutionPlan, ExecutionPlanItem, PlanAction
from feverslop.domain.prepared_workflow import SceneWorkflowManifest
from feverslop.domain.project_render_settings import ProjectRenderSettings
from feverslop.errors import FeverSlopDataError
from feverslop.scene_artifacts import SceneArtifactLayout


_H3_PIPELINES = frozenset({
    "minimax-h3-r2v", "minimax-h3-t2v", "minimax-h3-i2v",
    "minimax-h3-fl2v", "minimax-h3-l2v",
})
_LTX_PIPELINES = frozenset({"ltx_msr", "ltx_ingredients"})
_REFERENCE_PIPELINES = _LTX_PIPELINES | frozenset({"minimax-h3-r2v"})
_GLOBAL_ASSEMBLY_STAGES = frozenset({
    "concat_video_only",
    "mux_original_audio",
    "diagnostic_scene_audio_concat",
    "facefix_concat",
    "export_timeline",
    "openshot_export",
})


def build_resume_plan(
    project: str | Path,
    *,
    video_pipeline: str,
    selected_scenes: Iterable[int] | None = None,
    render_settings: ProjectRenderSettings | None = None,
) -> ExecutionPlan:
    root = Path(project).resolve()
    try:
        return _build_resume_plan(
            root,
            video_pipeline=video_pipeline,
            selected_scenes=selected_scenes,
            render_settings=render_settings,
        )
    except (FeverSlopDataError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return ExecutionPlan(root, "resume", (
            ExecutionPlanItem(
                "canonical plan",
                PlanAction.BLOCKED,
                f"canonical/provenance data is invalid; run `uv run python main.py plan validate {root}`",
            ),
        ))


def _build_resume_plan(
    project: Path,
    *,
    video_pipeline: str,
    selected_scenes: Iterable[int] | None = None,
    render_settings: ProjectRenderSettings | None = None,
) -> ExecutionPlan:
    root = project
    layout = SceneArtifactLayout(root)
    if not layout.base_plan.is_file():
        return ExecutionPlan(root, "resume", (
            ExecutionPlanItem("canonical plan", PlanAction.RUN, "canonical plan missing", stage="main_pipeline"),
        ))

    migration = analyze_canonical_plan_migration(CanonicalPlanStore(root).load())
    if migration.unresolved or migration.importable:
        count = len(migration.unresolved) + len(migration.importable)
        return ExecutionPlan(root, "resume", (
            ExecutionPlanItem(
                "canonical plan",
                PlanAction.BLOCKED,
                f"{count} legacy edit/provenance finding(s); run `uv run python main.py plan-migrate {root} --apply`",
            ),
        ))

    base = _read_plan(layout.base_plan)
    desired_base = (
        render_settings.apply_to_scenes(base)
        if render_settings is not None
        else base
    )
    requested = None if selected_scenes is None else {int(scene) for scene in selected_scenes}
    available = {int(scene["scene"]) for scene in desired_base}
    unknown = sorted((requested or set()) - available)
    if unknown:
        numbers = ", ".join(str(scene) for scene in unknown)
        return ExecutionPlan(root, "resume", (
            ExecutionPlanItem(
                "scene selection",
                PlanAction.BLOCKED,
                f"selected scene(s) not present in canonical plan: {numbers}",
            ),
        ))
    if desired_base != base and requested is not None and requested != available:
        return ExecutionPlan(root, "resume", (
            ExecutionPlanItem(
                "project render settings",
                PlanAction.BLOCKED,
                "global resolution or workflow changes require all scenes; rerun without --scenes",
            ),
        ))
    active_path = _active_plan(layout, video_pipeline)
    active = _read_plan(active_path) if active_path.is_file() else []
    stored_by_number = {int(scene["scene"]): scene for scene in active}
    reference_plan = _read_plan(layout.references_plan) if layout.references_plan.is_file() else []
    references_by_number = {int(scene["scene"]): scene for scene in reference_plan}
    items: list[ExecutionPlanItem] = []
    if desired_base != base:
        items.append(ExecutionPlanItem(
            "project render settings",
            PlanAction.RUN,
            _settings_change_reason(base, desired_base),
            stage="sync_project_settings",
        ))
    any_render = False

    for canonical_scene in desired_base:
        number = int(canonical_scene["scene"])
        if requested is not None and number not in requested:
            items.append(ExecutionPlanItem("scene", PlanAction.NOT_SELECTED, "outside --scenes selection", number))
            continue
        stored = stored_by_number.get(number)
        source = stored or canonical_scene
        current = project_effective_plan([source], desired_base)[0]
        current_dependencies = _dependencies(current)
        changed = _dependency_changes(stored, current)
        legacy_h3_workflow_reusable = (
            video_pipeline in _H3_PIPELINES
            and layout.scene_final_video(number).is_file()
            and not _workflow_mismatches(
                root,
                layout,
                number,
                video_pipeline,
                current_dependencies,
                allow_legacy_provenance=True,
            )
        )
        stored_reference = references_by_number.get(number)
        reference_changed = (
            stored_reference is None
            or _reference_inputs(stored_reference) != _reference_inputs(canonical_scene)
        )
        current_reference = project_effective_plan([stored_reference or canonical_scene], desired_base)[0]
        msr_prompts_fresh = _msr_prompts_fresh(stored_reference, current_reference)

        h3_action = PlanAction.REUSE
        if video_pipeline in _H3_PIPELINES:
            h3_action, h3_reason = _h3_state(layout, canonical_scene, number)
            if (
                reference_changed
                and video_pipeline in _REFERENCE_PIPELINES
                and (stored_reference is not None or desired_base != base)
            ):
                h3_action = PlanAction.RUN
                h3_reason = "reference generator or bindings changed"
            items.append(ExecutionPlanItem("h3 prompts", h3_action, h3_reason, number, "h3_prompts"))

        projection_stage = _projection_stage(video_pipeline)
        projection_action = PlanAction.REUSE
        projection_reason = "effective projection fingerprint matches"
        if projection_stage and (stored is None or changed) and not legacy_h3_workflow_reusable:
            projection_action = PlanAction.RUN
            projection_reason = "derived plan missing" if stored is None else "; ".join(changed)
        if projection_stage:
            if video_pipeline in _REFERENCE_PIPELINES:
                reference_inputs = canonical_scene.get("references") or {}
                reference_assets_reusable = (
                    not reference_changed
                    or reference_manifests_reusable(
                        root / "output" / "references",
                        actor_ids=reference_inputs.get("actor_ids", ()),
                        location_id=reference_inputs.get("location_id"),
                    )
                )
                items.append(ExecutionPlanItem(
                    "reference assets",
                    PlanAction.REUSE if reference_assets_reusable else PlanAction.RUN,
                    "existing reference manifests reusable"
                    if reference_assets_reusable and reference_changed
                    else "reference bindings changed or missing"
                    if reference_changed
                    else "reference assets reusable",
                    number,
                    "msr_references",
                ))
            if reference_changed and video_pipeline in _REFERENCE_PIPELINES:
                items.append(ExecutionPlanItem(
                    "reference bindings", PlanAction.RUN, projection_reason,
                    number, "msr_reference_sheets",
                ))
            else:
                items.append(ExecutionPlanItem(
                    "reference bindings", PlanAction.REUSE, "reference fingerprint matches",
                    number, "msr_reference_sheets",
                ))
            if video_pipeline == "ltx_ingredients":
                items.append(ExecutionPlanItem(
                    "MSR prompt enrichment",
                    PlanAction.REUSE if msr_prompts_fresh else PlanAction.RUN,
                    "MSR prompt input fingerprint matches" if msr_prompts_fresh else "MSR prompt inputs changed or provenance missing",
                    number,
                    "msr_prompt_enrich",
                ))
            elif video_pipeline == "ltx_msr":
                projection_action = PlanAction.REUSE if msr_prompts_fresh else PlanAction.RUN
                projection_reason = (
                    "MSR prompt input fingerprint matches"
                    if msr_prompts_fresh
                    else "MSR prompt inputs changed or provenance missing"
                )
            items.append(ExecutionPlanItem(
                "effective projection", projection_action, projection_reason,
                number, projection_stage,
            ))

        prepare_action = PlanAction.REUSE
        prepare_reason = "prepared workflow fingerprints match"
        if video_pipeline in _LTX_PIPELINES:
            mismatches = _workflow_mismatches(root, layout, number, video_pipeline, current_dependencies)
            if projection_action is PlanAction.RUN or mismatches:
                prepare_action = PlanAction.RUN
                prepare_reason = projection_reason if projection_action is PlanAction.RUN else "; ".join(mismatches)
            items.append(ExecutionPlanItem(
                "prepare", prepare_action, prepare_reason, number, "ltx_prepare_workflows",
            ))

        clip = layout.scene_final_video(number)
        render_mismatches: list[str] = []
        if video_pipeline not in _LTX_PIPELINES and clip.is_file():
            if not layout.scene_workflow(number).is_file() or not layout.scene_manifest(number).is_file():
                render_mismatches = ["render dependency provenance missing"]
            else:
                render_mismatches = _workflow_mismatches(
                    root, layout, number, video_pipeline, current_dependencies,
                    allow_legacy_provenance=video_pipeline in _H3_PIPELINES,
                )
        render_action = PlanAction.REUSE
        render_reason = "rendered clip exists and dependencies match"
        if (
            h3_action is PlanAction.RUN
            or projection_action is PlanAction.RUN
            or prepare_action is PlanAction.RUN
            or render_mismatches
            or not clip.is_file()
        ):
            render_action = PlanAction.RUN
            render_reason = (
                "clip missing" if not clip.is_file()
                else "; ".join(render_mismatches) if render_mismatches
                else "upstream scene dependency changed"
            )
            any_render = True
        items.append(ExecutionPlanItem(
            "render", render_action, render_reason, number, "ltx_render_scenes",
        ))

    partial_selection = requested is not None and len(requested) < len(desired_base)
    assembly_run = any_render or not layout.movie.is_file()
    assembly_action = PlanAction.RUN if assembly_run and not partial_selection else PlanAction.REUSE
    assembly_reason = (
        "partial scene selection; final assembly deferred"
        if partial_selection
        else "scene render changed"
        if any_render
        else "final movie missing"
        if assembly_run
        else "final movie exists"
    )
    for phase, stage in (
        ("assemble video", "concat_video_only"),
        ("mux audio", "mux_original_audio"),
        ("export timeline", "export_timeline"),
    ):
        items.append(ExecutionPlanItem(phase, assembly_action, assembly_reason, stage=stage))
    return ExecutionPlan(root, "resume", tuple(items))


def _settings_change_reason(
    before: list[Mapping[str, Any]],
    after: list[Mapping[str, Any]],
) -> str:
    reasons: list[str] = []
    for old, new in zip(before, after, strict=True):
        if (old.get("width"), old.get("height")) != (new.get("width"), new.get("height")):
            reasons.append("resolution changed")
        if old.get("render_settings") != new.get("render_settings"):
            reasons.append("video workflow changed")
        old_generator = (old.get("references") or {}).get("generator_fingerprint")
        new_generator = (new.get("references") or {}).get("generator_fingerprint")
        if old_generator != new_generator:
            reasons.append("reference workflow changed")
    return "; ".join(dict.fromkeys(reasons)) or "project render settings changed"


def build_compatibility_plan(
    project: str | Path,
    stages: Iterable[str],
    *,
    selected_scenes: Iterable[int] | None = None,
) -> ExecutionPlan:
    scenes = tuple(sorted(set(int(scene) for scene in (selected_scenes or ()))))
    items = tuple(
        ExecutionPlanItem(
            "advanced stage",
            (
                PlanAction.REUSE
                if scenes and str(stage) in _GLOBAL_ASSEMBLY_STAGES
                else PlanAction.RUN
            ),
            (
                "partial scene selection; global assembly deferred"
                if scenes and str(stage) in _GLOBAL_ASSEMBLY_STAGES
                else "selected by compatibility flags"
            ),
            None,
            str(stage),
        )
        for stage in stages
    )
    if scenes:
        items += tuple(
            ExecutionPlanItem("scene selection", PlanAction.RUN, "selected by --scenes", scene)
            for scene in scenes
        )
    return ExecutionPlan(Path(project).resolve(), "compatibility", items)


def _active_plan(layout: SceneArtifactLayout, pipeline: str) -> Path:
    if pipeline == "ltx_msr":
        return layout.references_plan
    if pipeline == "ltx_ingredients":
        return layout.ingredients_plan
    return layout.base_plan


def _read_plan(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list) or any(not isinstance(scene, dict) for scene in payload):
        raise ValueError(f"Render plan must be a list of objects: {path}")
    return payload


def _dependencies(scene: Mapping[str, Any]) -> CanonicalSceneDependencies:
    payload = ((scene.get("canonical_projection") or {}).get("dependencies"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Scene {scene.get('scene')} has no canonical dependency projection")
    return CanonicalSceneDependencies.from_dict(payload)


def _dependency_changes(stored: Mapping[str, Any] | None, current: Mapping[str, Any]) -> list[str]:
    if stored is None:
        return ["derived plan missing"]
    before = ((stored.get("canonical_projection") or {}).get("dependencies"))
    after = ((current.get("canonical_projection") or {}).get("dependencies"))
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return ["canonical projection provenance missing"]
    changed = []
    if before.get("reference_fingerprint") != after.get("reference_fingerprint"):
        changed.append("reference fingerprint changed")
    if before.get("workflow_fingerprint") != after.get("workflow_fingerprint"):
        changed.append("workflow fingerprint changed")
    return changed


def _reference_inputs(scene: Mapping[str, Any]) -> Any:
    """Return only user/config-owned reference inputs, excluding generated paths."""
    return _strip_derived_reference_values(scene.get("references"))


def reference_manifests_reusable(
    references_dir: Path,
    *,
    actor_ids: Iterable[str],
    location_id: str | Iterable[str] | None,
) -> bool:
    """Return whether all configured MSR references have usable local sheets."""
    required = [("actors", str(identifier)) for identifier in actor_ids if str(identifier).strip()]
    locations = (location_id,) if isinstance(location_id, str) else (location_id or ())
    required.extend(
        ("locations", str(identifier))
        for identifier in locations
        if str(identifier).strip()
    )
    if not required:
        return False
    return all(_reference_manifest_reusable(references_dir, kind, identifier) for kind, identifier in required)


def _reference_manifest_reusable(references_dir: Path, kind: str, identifier: str) -> bool:
    manifest_path = references_dir / kind / identifier / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, TypeError, ValueError):
        return False
    if not isinstance(manifest, Mapping):
        return False
    paths = [manifest.get("sheet_path"), manifest.get("msr_input_path"), manifest.get("sequence_sheet_path")]
    for raw_path in paths:
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        candidate = Path(raw_path)
        for resolved in (
            manifest_path.parent / candidate,
            references_dir / candidate,
            references_dir.parent.parent / candidate,
        ):
            if resolved.is_file():
                return True
    return any((manifest_path.parent / name).is_file() for name in ("sheet.png", "msr_sheet.png"))


def _strip_derived_reference_values(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_derived_reference_values(item)
            for key, item in value.items()
            if str(key) != "_stem_audio_tags"
            and not any(marker in str(key).lower() for marker in ("path", "sha", "sheet", "anchor"))
        }
    if isinstance(value, list):
        return [_strip_derived_reference_values(item) for item in value]
    return value


def _msr_prompts_fresh(
    stored: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> bool:
    if stored is None:
        return False
    provenance = (stored.get("stage_provenance") or {}).get("msr_prompt_enrich")
    if not isinstance(provenance, Mapping):
        return False
    return provenance.get("input_fingerprint") == msr_prompt_input_fingerprint(dict(current))


def _projection_stage(pipeline: str) -> str | None:
    if pipeline == "ltx_msr":
        return "msr_prompt_enrich"
    if pipeline == "ltx_ingredients":
        return "ingredients_sheets"
    if pipeline in _H3_PIPELINES:
        return "render_plan"
    return None


def _h3_state(layout: SceneArtifactLayout, scene: Mapping[str, Any], number: int) -> tuple[PlanAction, str]:
    role = (((scene.get("canonical") or {}).get("roles") or {}).get(str(PromptRole.H3_VIDEO)) or {})
    override = role.get("override")
    if isinstance(override, Mapping):
        return PlanAction.REUSE, "human H3 override is authoritative"
    checkpoint = layout.scene_h3_prompt(number)
    if not checkpoint.is_file():
        return PlanAction.RUN, "judged H3 checkpoint missing"
    payload = json.loads(checkpoint.read_text(encoding="utf-8-sig"))
    expected = ((role.get("generated") or {}).get("provenance") or {}).get("input_fingerprint")
    if expected and expected != payload.get("input_fingerprint"):
        return PlanAction.RUN, "H3 input fingerprint changed"
    if str(payload.get("status") or "").lower() not in {"good", "unjudged"}:
        return PlanAction.RUN, "H3 checkpoint is not renderable"
    return PlanAction.REUSE, "judged H3 checkpoint matches"


def _workflow_mismatches(
    project: Path,
    layout: SceneArtifactLayout,
    scene: int,
    pipeline: str,
    dependencies: CanonicalSceneDependencies,
    *,
    allow_legacy_provenance: bool = False,
) -> list[str]:
    workflow = layout.scene_workflow(scene)
    manifest_path = layout.scene_manifest(scene)
    if not workflow.is_file() or not manifest_path.is_file():
        return ["prepared workflow missing"]
    manifest = SceneWorkflowManifest.read(manifest_path)
    mismatches = []
    if manifest.pipeline != pipeline:
        mismatches.append("prepared workflow pipeline changed")
    dependency_mismatches = manifest.compare_canonical_dependencies(dependencies)
    if allow_legacy_provenance:
        dependency_mismatches = [
            mismatch for mismatch in dependency_mismatches
            if mismatch != "canonical dependency provenance is missing"
        ]
    mismatches.extend(dependency_mismatches)
    mismatches.extend(manifest.verify(project))
    return mismatches
