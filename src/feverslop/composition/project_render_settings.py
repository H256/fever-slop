from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from feverslop.adapters.pipeline_runner_options import RUNNER_ARGUMENTS
from feverslop.config.project_config import ProjectConfig
from feverslop.domain.project_render_settings import (
    ProjectRenderSettings,
    WorkflowSelection,
)

from .config_loader import resolve_runner_path, runner_root


@dataclass(frozen=True)
class ResolvedProjectRenderSettings:
    settings: ProjectRenderSettings
    runner_overrides: dict[str, str]


def resolve_project_render_settings(
    project: str | Path,
    *,
    video_pipeline: str,
    explicit_runner_options: Collection[str] = (),
    reference_generation: str | None = None,
    sequence_to_sheet_workflow: str | None = None,
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
    if config.workflows.video is None and config.render_profile.startswith("ltx25-"):
        profile_parts = config.render_profile.split("-")
        if len(profile_parts) == 3 and profile_parts[1] in {"t2v", "i2v", "r2v", "msr", "ingredients"} and profile_parts[2] in {"draft", "standard", "final"}:
            mode, quality = profile_parts[1], profile_parts[2]
            profile_path = runner_root() / "workflows" / "video" / "ltx_25" / mode / f"{mode}_{quality}.json"
            if profile_path.exists() and video_target not in explicit:
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
