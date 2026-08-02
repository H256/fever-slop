from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from feverslop.scene_artifacts import SceneArtifactLayout
from feverslop.studio.pipeline_state_store import reconcile_completed_stages


def pipeline_action_availability(project_root: Path, scenes: list[int] | None = None) -> list[dict[str, Any]]:
    """Return Studio pipeline actions with their current prerequisites.

    LTX rendering is intentionally gated by both materialized files per selected
    scene.  The runner makes the same requirement; exposing it here prevents a
    job that can only fail before it reaches ComfyUI.
    """
    selected_scenes = sorted(set(scenes or []))
    completed = _completed_actions(project_root)
    main_ready = _complete(completed, "main-pipeline", "main_pipeline", "full-pipeline")
    references_ready = _complete(completed, "msr-references", "msr_references", "full-pipeline")
    enrichment_ready = _complete(completed, "msr-enrich", "msr_prompt_enrich", "full-pipeline")
    missing_workflows = _missing_workflow_scenes(project_root, selected_scenes)
    selection_reason = "Select at least one scene first."
    render_reason = (
        selection_reason
        if not selected_scenes
        else _prepare_reason(missing_workflows)
    )
    render_enabled = bool(selected_scenes) and not missing_workflows
    prepare_enabled = bool(selected_scenes) and enrichment_ready

    return [
        _action("Full pipeline", "full-pipeline"),
        _action("Main pipeline", "main-pipeline", recommended=not main_ready),
        _action(
            "MSR references", "msr-references", enabled=main_ready,
            recommended=main_ready and not references_ready,
            reason="Run Main pipeline first." if not main_ready else "",
        ),
        _action(
            "MSR enrichment", "msr-enrich", enabled=references_ready,
            recommended=references_ready and not enrichment_ready,
            reason="Run MSR references first." if not references_ready else "",
        ),
        _action(
            "Prepare LTX workflows",
            "ltx-prepare-workflows",
            enabled=prepare_enabled,
            recommended=prepare_enabled and bool(missing_workflows),
            reason=(
                selection_reason if not selected_scenes
                else "Run MSR enrichment first." if not enrichment_ready
                else ""
            ),
        ),
        _action(
            "Render selected scenes",
            "ltx-render-scenes",
            enabled=render_enabled,
            recommended=render_enabled,
            reason=render_reason,
        ),
        _action(
            "Final concat", "final-concat",
            enabled=_all_render_plan_clips_exist(project_root),
            reason="" if _all_render_plan_clips_exist(project_root) else "Render scene clips first.",
        ),
    ]


def ensure_pipeline_action_available(project_root: Path, action: str, scenes: list[int] | None = None) -> None:
    """Reject an unavailable Studio action at the service boundary."""
    actions = {item["value"]: item for item in pipeline_action_availability(project_root, scenes)}
    item = actions.get(action)
    if item is not None and not item["enabled"]:
        raise ValueError(str(item["reason"]))


def _missing_workflow_scenes(project_root: Path, scenes: list[int]) -> list[int]:
    missing: list[int] = []
    for scene in scenes:
        scene_dir = project_root / "output" / "render" / "scenes" / f"scene_{scene:04d}"
        if not (scene_dir / "workflow.json").is_file() or not (scene_dir / "manifest.json").is_file():
            missing.append(scene)
    return missing


def _all_render_plan_clips_exist(project_root: Path) -> bool:
    layout = SceneArtifactLayout(project_root)
    video_pipeline = _configured_video_pipeline(project_root)
    candidates = (
        (layout.references_plan, layout.base_plan)
        if video_pipeline == "ltx_msr"
        else (layout.ingredients_plan, layout.base_plan)
        if video_pipeline == "ltx_ingredients"
        else (layout.base_plan,)
    )
    render_plan = next(
        (path for path in candidates if path.is_file()),
        None,
    )
    if render_plan is None:
        return False
    try:
        scenes = json.loads(render_plan.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(scenes, list) or not scenes:
        return False
    legacy_dirs = (layout.render_dir / "ltx_msr", layout.render_dir / "ltx_ingredients")
    try:
        return all(
            isinstance(scene, dict)
            and layout.find_scene_final_video(int(scene["scene"]), legacy_dirs=legacy_dirs) is not None
            for scene in scenes
        )
    except (KeyError, TypeError, ValueError):
        return False


def _configured_video_pipeline(project_root: Path) -> str:
    try:
        config = json.loads((project_root / "config.json").read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(config.get("video_pipeline") or "") if isinstance(config, dict) else ""


def _completed_actions(project_root: Path) -> set[str]:
    path = project_root / ".studio" / "pipeline_state.json"
    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(payload, dict):
        return set()
    return set(reconcile_completed_stages(payload))


def _complete(completed: set[str], *actions: str) -> bool:
    return any(action in completed for action in actions)


def _prepare_reason(scenes: list[int]) -> str:
    if not scenes:
        return ""
    numbers = ", ".join(str(scene) for scene in scenes)
    return f"Prepare LTX workflows for scenes {numbers} first."


def _action(
    label: str,
    value: str,
    *,
    enabled: bool = True,
    recommended: bool = False,
    reason: str = "",
) -> dict[str, Any]:
    return {
        "label": label,
        "value": value,
        "enabled": enabled,
        "recommended": recommended,
        "reason": reason,
    }
