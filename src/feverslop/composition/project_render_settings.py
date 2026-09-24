from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from feverslop.adapters.pipeline_runner_options import RUNNER_ARGUMENTS
from feverslop.config.project_config import ProjectConfig
from feverslop.domain.project_render_settings import (
    ProjectRenderSettings,
    WorkflowSelection,
)
from feverslop.domain.render_profile import (
    RegisteredRenderProfile,
    RenderProfile,
    RenderProfileRegistry,
)
from feverslop.domain.workflow_capability_manifest import WorkflowCapabilityManifest
from feverslop.domain.ltx25_audio_contract import (
    load_ltx25_audio_policy,
    validate_ltx25_audio_workflow,
)

from .config_loader import resolve_runner_path, runner_root

#: H3 render pipelines all map to the single ``minimax_h3`` import-store family.
_H3_FAMILY_PIPELINES = frozenset({"minimax-h3-r2v", "minimax-h3-t2v", "minimax-h3-i2v"})


@dataclass(frozen=True)
class ResolvedProjectRenderSettings:
    settings: ProjectRenderSettings
    runner_overrides: dict[str, str]


def _import_store_for(project: Path, app_config: Any) -> Any:
    """Anchor and return the per-project workflow-import store, or None.

    ``app_config`` may be a loaded :class:`AppConfig` (re-anchored to the
    project dir) or ``None`` (a default config is used). Returns ``None``
    when the project has no ``workflows`` directory, so the caller falls
    back to the built-in default.
    """
    from feverslop.config.app_config import AppConfig, ComfyUIConfig, LLMConfig

    config = app_config if app_config is not None else AppConfig(llm=LLMConfig(), comfyui=ComfyUIConfig())
    config.attach_import_store(project)
    return config.import_store


def _active_import_path(project: Path, app_config: Any, video_pipeline: str) -> str | None:
    """Return the active H3 import snapshot path, or None to use the built-in default.

    All H3 render pipelines map to the single ``minimax_h3`` import-store
    family. An active ``final`` import (validated + test-run + activated)
    takes precedence over the built-in two-pass workflow, mirroring the LTX
    import-precedence path.
    """
    if video_pipeline not in _H3_FAMILY_PIPELINES:
        return None
    store = _import_store_for(project, app_config)
    if store is None:
        return None
    active = store.find_active(pipeline="minimax_h3", purpose="final")
    if active is None:
        return None
    return str(store.snapshot_path(active.profile_id))


def _load_declared_ltx25_profiles() -> RenderProfileRegistry:
    """Resolve the canonical LTX 2.5 profile matrix into a registry.

    The profile-matrix.json is the single source of truth for which LTX 2.5
    profiles are declared. Each entry maps to its materialized workflow file.
    """
    matrix_path = runner_root() / "workflows" / "video" / "ltx_25" / "profile-matrix.json"
    entries = json.loads(matrix_path.read_text(encoding="utf-8"))
    registered = []
    for entry in entries:
        profile = RenderProfile.create(model_family="ltx-2.5", **entry)
        registered.append(
            RegisteredRenderProfile(
                profile=profile,
                workflow_path=(
                    "workflows/video/ltx_25/"
                    f"{profile.mode.value}/{profile.mode.value}_{profile.quality.value}.json"
                ),
            )
        )
    return RenderProfileRegistry(registered)


def resolve_project_render_settings(
    project: str | Path,
    *,
    video_pipeline: str,
    explicit_runner_options: Collection[str] = (),
    reference_generation: str | None = None,
    sequence_to_sheet_workflow: str | None = None,
    app_config: Any = None,
) -> ResolvedProjectRenderSettings:
    root = Path(project).resolve()
    config = ProjectConfig.load(root / "config.json")
    explicit = frozenset(explicit_runner_options)
    overrides: dict[str, str] = {}
    effective_reference_generation = reference_generation or config.reference_generation
    if "reference_generation" not in explicit:
        overrides["reference_generation"] = effective_reference_generation
    configured_sequence_workflow = (
        sequence_to_sheet_workflow
        or config.workflows.reference_sequence
        or next(
            kwargs.get("default")
            for name, _flags, kwargs in RUNNER_ARGUMENTS
            if name == "sequence_to_sheet_workflow"
        )
    )
    if config.workflows.reference_sequence is not None and "sequence_to_sheet_workflow" not in explicit:
        sequence_path = resolve_runner_path(configured_sequence_workflow).resolve()
        overrides["sequence_to_sheet_workflow"] = str(sequence_path)
    video_selection = None
    video_target = {
        "ltx_msr": "msr_workflow",
        "ltx_ingredients": "ingredients_workflow",
    }.get(video_pipeline, "single_prompt_workflow")
    pipeline_defaults = {
        "minimax-h3-r2v": "workflows/video/minimax_h3/r2v_audio_two_pass.json",
        "minimax-h3-t2v": "workflows/video/minimax_h3/t2v_two_pass.json",
        "minimax-h3-i2v": "workflows/video/minimax_h3/i2v_two_pass.json",
    }
    pipeline_default = pipeline_defaults.get(video_pipeline)
    if pipeline_default and config.workflows.video is None and video_target not in explicit:
        # An active, validated H3 import (validated + test-run + activated)
        # takes precedence over the built-in two-pass workflow, mirroring the
        # LTX import-precedence path. Fall back to the built-in default when
        # no import exists.
        import_path = _active_import_path(root, app_config, video_pipeline)
        if import_path is not None:
            profile_path = Path(import_path).resolve()
        else:
            profile_path = resolve_runner_path(pipeline_default).resolve()
        video_selection = WorkflowSelection.from_path(profile_path, root=runner_root())
        overrides[video_target] = str(profile_path)
    if config.workflows.video is None and not pipeline_default and config.render_profile.startswith("ltx25-"):
        if video_target not in explicit:
            registry = _load_declared_ltx25_profiles()
            entry = registry.resolve(profile_id=config.render_profile)
            profile_path = runner_root() / entry.workflow_path
            if not profile_path.is_file():
                raise ValueError(
                    f"LTX 2.5 profile {config.render_profile!r} is declared but its "
                    f"workflow file is missing: {entry.workflow_path}"
                )
            manifest_path = runner_root() / "workflows" / "video" / "ltx_25" / "capabilities.json"
            manifest = WorkflowCapabilityManifest.create(
                **json.loads(manifest_path.read_text(encoding="utf-8"))
            )
            validation = manifest.validate_workflow_payload(
                json.loads(profile_path.read_text(encoding="utf-8-sig"))
            )
            if not validation.ok:
                missing = ", ".join((*validation.missing_models, *validation.missing_nodes))
                raise ValueError(f"LTX 2.5 workflow capability validation failed: {missing}")
            audio_policy = load_ltx25_audio_policy(profile_path)
            validate_ltx25_audio_workflow(
                json.loads(profile_path.read_text(encoding="utf-8-sig")),
                audio_policy,
            )
            video_selection = WorkflowSelection.from_path(profile_path.resolve(), root=runner_root())
            overrides[video_target] = str(profile_path.resolve())
    if config.workflows.video is not None and video_target not in explicit:
        video_path = resolve_runner_path(config.workflows.video).resolve()
        video_selection = WorkflowSelection.from_path(video_path, root=runner_root())
        overrides[video_target] = str(video_path)

    hero_selection = None
    edit_selection = None
    if config.workflows.reference_hero is not None or config.workflows.reference_edit is not None:
        defaults = {
            name: kwargs.get("default")
            for name, _flags, kwargs in RUNNER_ARGUMENTS
        }
        reference_names = {"reference_hero_workflow", "reference_edit_workflow"}
        resolve_complete_pair = explicit.isdisjoint(reference_names)
        if "reference_hero_workflow" not in explicit and (
            resolve_complete_pair or config.workflows.reference_hero is not None
        ):
            hero_path = resolve_runner_path(
                config.workflows.reference_hero or defaults["reference_hero_workflow"],
            ).resolve()
            hero_selection = WorkflowSelection.from_path(hero_path, root=runner_root())
            overrides["reference_hero_workflow"] = str(hero_path)
        if "reference_edit_workflow" not in explicit and (
            resolve_complete_pair or config.workflows.reference_edit is not None
        ):
            edit_path = resolve_runner_path(
                config.workflows.reference_edit or defaults["reference_edit_workflow"],
            ).resolve()
            edit_selection = WorkflowSelection.from_path(edit_path, root=runner_root())
            overrides["reference_edit_workflow"] = str(edit_path)

    return ResolvedProjectRenderSettings(
        settings=ProjectRenderSettings(
            width=config.video.width,
            height=config.video.height,
            video_workflow=video_selection,
            reference_hero_workflow=hero_selection,
            reference_edit_workflow=edit_selection,
            reference_generation=effective_reference_generation,
            reference_sequence_workflow=(
                WorkflowSelection.from_path(
                    resolve_runner_path(configured_sequence_workflow).resolve(),
                    root=runner_root(),
                )
                if effective_reference_generation == "sequence_sheet"
                else None
            ),
        ),
        runner_overrides=overrides,
    )
