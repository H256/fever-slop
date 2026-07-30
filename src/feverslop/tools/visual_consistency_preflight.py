from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

from feverslop.adapters.project_visual_consistency import (
    ProjectReferenceManifestAdapter,
    validate_project_scene_artifacts,
)
from feverslop.application.visual_consistency_preflight import (
    VisualConsistencyPreflightResult,
    preflight_visual_consistency,
)
from feverslop.config.project_config import ProjectConfig
from feverslop.domain.visual_consistency import PreflightMode


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only visual consistency validation for a render plan."
    )
    parser.add_argument("project_dir")
    parser.add_argument(
        "--plan",
        default="output/render/plans/ingredients.json",
        help="Render plan path inside PROJECT_DIR.",
    )
    parser.add_argument("--mode", choices=("ingredients", "msr", "i2v"), default="ingredients")
    parser.add_argument(
        "--preflight-mode",
        type=PreflightMode.parse,
        choices=tuple(PreflightMode),
        default=PreflightMode.WARN,
    )
    parser.add_argument("--workflow-profile", default=None)
    parser.add_argument("--json", action="store_true")
    return parser


def run(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        project = Path(args.project_dir).resolve(strict=True)
        if not project.is_dir():
            raise ValueError(f"Project is not a directory: {project}")
        plan_path = Path(args.plan)
        plan_path = (
            plan_path.resolve()
            if plan_path.is_absolute()
            else (project / plan_path).resolve()
        )
        if not plan_path.is_relative_to(project):
            raise ValueError("Plan path must be inside the project")
        payload = json.loads(plan_path.read_text(encoding="utf-8-sig"))
        scenes = _plan_scenes(payload)
        if args.preflight_mode is PreflightMode.OFF:
            result = VisualConsistencyPreflightResult((), ())
        else:
            config_path = project / "config.json"
            config = ProjectConfig.load(config_path) if config_path.exists() else None
            snapshot = ProjectReferenceManifestAdapter(
                lambda _project_id: project
            ).load(project.name)
            result = preflight_visual_consistency(
                scenes,
                snapshot,
                mode=args.mode,
                workflow_profile=args.workflow_profile or f"{args.mode}-default",
                preflight_mode=args.preflight_mode,
                subject_mode=config.subject_mode if config else "multi",
                max_scene_actors=config.max_scene_actors if config else 4,
                supports_continuous_transitions=args.mode == "i2v",
            )
            artifact_issues = validate_project_scene_artifacts(
                project,
                scenes,
                mode=args.mode,
                preflight_mode=args.preflight_mode,
            )
            result = VisualConsistencyPreflightResult(
                result.contracts,
                (*result.issues, *artifact_issues),
            )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        _print_error(str(exc), json_output=args.json)
        return 1

    _print_result(result, json_output=args.json)
    return 0 if result.renderable else 2


def _plan_scenes(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        scenes = payload
    elif isinstance(payload, dict):
        scenes = payload.get("scenes", payload.get("shots"))
    else:
        scenes = None
    if not isinstance(scenes, list) or not all(
        isinstance(scene, dict) for scene in scenes
    ):
        raise ValueError("Render plan must be a JSON array or contain scenes/shots")
    return scenes


def _print_result(
    result: VisualConsistencyPreflightResult,
    *,
    json_output: bool,
) -> None:
    payload = {
        "renderable": result.renderable,
        "contracts": [contract.to_dict() for contract in result.contracts],
        "issues": [asdict(issue) for issue in result.issues],
    }
    if json_output:
        print(json.dumps(payload, sort_keys=True))
        return
    state = "renderable" if result.renderable else "blocked"
    print(
        f"Visual consistency preflight: {state}; "
        f"{len(result.contracts)} contract(s), {len(result.issues)} issue(s)"
    )
    for issue in result.issues:
        print(f"[{issue.severity}] scene {issue.scene} {issue.code}: {issue.message}")


def _print_error(message: str, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps({"error": message}))
    else:
        print(f"Could not run visual consistency preflight: {message}")


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
