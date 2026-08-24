from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from feverslop.adapters.canonical_plan_store import CanonicalPlanStore
from feverslop.application.canonical_plan_inspection import inspect_overrides, inspect_scene_roles
from feverslop.application.canonical_plan_migration import analyze_canonical_plan_migration
from feverslop.application.effective_render_plan import project_effective_plan
from feverslop.composition.project_render_settings import (
    resolve_project_render_settings,
)
from feverslop.domain.effective_render_plan import CanonicalSceneDependencies, canonical_plan_revision
from feverslop.domain.prepared_workflow import SceneWorkflowManifest
from feverslop.errors import FeverSlopDataError
from feverslop.scene_artifacts import SceneArtifactLayout


def build_canonical_plan_parsers(subparsers) -> None:
    plan = subparsers.add_parser("plan", help="Inspect the canonical render plan without modifying it.")
    commands = plan.add_subparsers(dest="plan_command", required=True)
    for name in ("path", "validate", "overrides"):
        command = commands.add_parser(name)
        command.add_argument("project")
        if name == "overrides":
            command.add_argument("--orphans", action="store_true")
    show = commands.add_parser("show")
    show.add_argument("project")
    show.add_argument("--scene", type=int, required=True)
    status = subparsers.add_parser("status", help="Show read-only canonical artifact freshness.")
    status.add_argument("project")


def run_canonical_plan_command(args: argparse.Namespace, *, console: Console | None = None) -> int:
    output = console or Console()
    project = Path(args.project).resolve()
    try:
        if args.command == "status":
            return _status(project, output)
        return {
            "path": _path,
            "validate": _validate,
            "show": _show,
            "overrides": _overrides,
        }[args.plan_command](project, args, output)
    except (FeverSlopDataError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        output.print(f"[red]Invalid/corrupt project:[/red] {exc}")
        return 1


def _load_base(project: Path) -> list[dict[str, Any]]:
    path = SceneArtifactLayout(project).base_plan
    if not path.is_file():
        raise FileNotFoundError(f"Canonical base plan does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"Canonical base plan must be a list of objects: {path}")
    return value


def _path(project: Path, _args: argparse.Namespace, output: Console) -> int:
    layout = SceneArtifactLayout(project)
    table = Table("Artifact", "Role", "State")
    table.add_row(str(layout.base_plan), "SOLE EDITABLE PLAN", "present" if layout.base_plan.is_file() else "MISSING")
    for path in (layout.compact_plan, layout.anchored_plan, layout.references_plan, layout.ingredients_plan):
        table.add_row(str(path), "derived cache", "present" if path.is_file() else "missing")
    output.print(table)
    return 0 if layout.base_plan.is_file() else 2


def _validate(project: Path, _args: argparse.Namespace, output: Console) -> int:
    report = analyze_canonical_plan_migration(CanonicalPlanStore(project).load())
    base = _load_base(project)
    for scene in base:
        inspect_scene_roles(base, int(scene.get("scene") or 0))
    revision = canonical_plan_revision(base)
    output.print(f"Canonical revision: {revision}")
    stale_projections = _stale_projection_identities(project, base)
    for finding in stale_projections:
        output.print(f"[yellow]ACTION REQUIRED[/yellow] {finding}")
    for finding in report.unresolved:
        output.print(f"[yellow]ACTION REQUIRED[/yellow] {finding.source_path}: {finding.reason}")
    if report.importable:
        output.print(f"[yellow]ACTION REQUIRED[/yellow] {len(report.importable)} legacy edit(s); run plan-migrate")
    if report.unresolved or report.importable or stale_projections:
        return 2
    output.print("[green]VALID[/green] canonical plan and provenance")
    return 0


def _show(project: Path, args: argparse.Namespace, output: Console) -> int:
    scene_id, roles = inspect_scene_roles(_load_base(project), args.scene)
    output.print(f"Scene {args.scene} | canonical.scene_id {scene_id}")
    table = Table("Role", "Owner", "Generated", "Override", "Effective", "Provenance")
    for role in roles:
        provenance = role.override_provenance if role.owner == "override" else role.generated_provenance
        table.add_row(role.role, role.owner, _display(role.generated), _display(role.override), _display(role.effective), _display(provenance))
    output.print(table)
    return 0


def _overrides(project: Path, args: argparse.Namespace, output: Console) -> int:
    rows = inspect_overrides(_load_base(project))
    table = Table("Scene", "Scene ID", "Role", "Provenance")
    for row in rows:
        table.add_row(str(row["scene"]), row["scene_id"], row["role"], _display(row["provenance"]))
    report = analyze_canonical_plan_migration(CanonicalPlanStore(project).load())
    orphans = [finding for finding in report.unresolved if "orphan" in finding.reason]
    if args.orphans:
        for finding in orphans:
            table.add_row(str(finding.scene_number or "?"), finding.scene_id or "unmatched", finding.role or "-", f"ORPHAN {finding.source_path}: {finding.reason}")
    output.print(table)
    return 2 if orphans else 0


def _status(project: Path, output: Console) -> int:
    layout = SceneArtifactLayout(project)
    if not layout.base_plan.is_file():
        output.print("[yellow]MISSING[/yellow] canonical plan — required next phase: main_pipeline")
        return 2
    base = _load_base(project)
    for scene in base:
        inspect_scene_roles(base, int(scene.get("scene") or 0))
    migration = analyze_canonical_plan_migration(CanonicalPlanStore(project).load())
    table = Table("Phase", "Scene", "State", "Cause / required next phase")
    action_required = False
    stored_base = base
    config_path = project / "config.json"
    if config_path.is_file():
        raw_config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        pipeline = str(raw_config.get("video_pipeline") or "ltx_i2v")
        settings = resolve_project_render_settings(
            project,
            video_pipeline=pipeline,
        ).settings
        base = settings.apply_to_scenes(base)
    if base != stored_base:
        reasons = []
        if any(
            (old.get("width"), old.get("height")) != (new.get("width"), new.get("height"))
            for old, new in zip(stored_base, base, strict=True)
        ):
            reasons.append("resolution changed")
        if any(old.get("render_settings") != new.get("render_settings") for old, new in zip(stored_base, base, strict=True)):
            reasons.append("video workflow changed")
        if any(
            (old.get("references") or {}).get("generator_fingerprint")
            != (new.get("references") or {}).get("generator_fingerprint")
            for old, new in zip(stored_base, base, strict=True)
        ):
            reasons.append("reference workflow changed")
        table.add_row(
            "canonical",
            "all",
            "STALE",
            f"{'; '.join(reasons)}; required next phase: sync_project_settings",
        )
        action_required = True
    else:
        table.add_row("canonical", "all", "READY", f"revision {canonical_plan_revision(base)[:12]}")
    if migration.unresolved or migration.importable:
        count = len(migration.unresolved) + len(migration.importable)
        table.add_row("legacy edits", "all", "BLOCKED", f"{count} finding(s); required next phase: plan-migrate")
        action_required = True
    active = next((path for path in (layout.ingredients_plan, layout.references_plan, layout.base_plan) if path.is_file()), layout.base_plan)
    derived = json.loads(active.read_text(encoding="utf-8-sig"))
    projected = project_effective_plan(derived, base)
    for stored_scene, scene in zip(derived, projected, strict=True):
        number = int(scene["scene"])
        stored_dependencies = ((stored_scene.get("canonical_projection") or {}).get("dependencies"))
        current_dependencies = ((scene.get("canonical_projection") or {}).get("dependencies"))
        if isinstance(stored_dependencies, dict) and isinstance(current_dependencies, dict):
            changed = []
            if stored_dependencies.get("workflow_fingerprint") != current_dependencies.get("workflow_fingerprint"):
                changed.append("workflow fingerprint changed")
            if stored_dependencies.get("reference_fingerprint") != current_dependencies.get("reference_fingerprint"):
                changed.append("reference fingerprint changed")
            if changed:
                next_phase = "ingredients_sheets or msr_reference_sheets" if "reference fingerprint changed" in changed else "ltx_prepare_workflows"
                table.add_row("derived plan", str(number), "STALE", f"{'; '.join(changed)}; required next phase: {next_phase}")
                action_required = True
            else:
                table.add_row("derived plan", str(number), "READY", "scene-local fingerprints match")
        elif active != layout.base_plan:
            table.add_row("derived plan", str(number), "PARTIAL", "provenance missing; required next phase: enrichment")
            action_required = True
        checkpoint = layout.scene_h3_prompt(number)
        if not checkpoint.is_file():
            table.add_row("h3 checkpoint", str(number), "PARTIAL", "missing; required next phase: h3_prompts")
            action_required = True
        else:
            payload = json.loads(checkpoint.read_text(encoding="utf-8-sig"))
            status = str(payload.get("status") or "BLOCKED").upper()
            canonical = scene.get("canonical") or {}
            role = (canonical.get("roles") or {}).get("h3.video") or {}
            expected = ((role.get("generated") or {}).get("provenance") or {}).get("input_fingerprint")
            fresh = not expected or expected == payload.get("input_fingerprint")
            state = "READY" if fresh and status in {"GOOD", "UNJUDGED"} else "STALE" if not fresh else "BLOCKED"
            table.add_row("h3 checkpoint", str(number), state, f"judge {status}; " + ("fresh" if fresh else "required next phase: h3_prompts"))
            action_required |= state != "READY"
        projection = scene.get("canonical_projection") or {}
        dependencies = projection.get("dependencies")
        manifest_path = layout.scene_manifest(number)
        workflow_path = layout.scene_workflow(number)
        if not manifest_path.is_file() or not workflow_path.is_file():
            table.add_row("prepared workflow", str(number), "MISSING", "required next phase: ltx_prepare_workflows")
            action_required = True
            continue
        manifest = SceneWorkflowManifest.read(manifest_path)
        current = CanonicalSceneDependencies.from_dict(dependencies) if isinstance(dependencies, dict) else None
        mismatches = manifest.compare_canonical_dependencies(current) if current is not None else ["canonical provenance missing"]
        mismatches.extend(manifest.verify(project))
        if mismatches:
            table.add_row("prepared workflow", str(number), "STALE", f"{'; '.join(mismatches)}; required next phase: ltx_prepare_workflows")
            action_required = True
        else:
            table.add_row("prepared workflow", str(number), "READY", "fingerprints and artifacts match")
    output.print(table)
    return 2 if action_required else 0


def _display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _stale_projection_identities(
    project: Path,
    base: list[dict[str, Any]],
) -> list[str]:
    layout = SceneArtifactLayout(project)
    expected = {
        int(scene.get("scene") or 0): str((scene.get("canonical") or {}).get("scene_id") or "")
        for scene in base
    }
    findings = []
    for path in (layout.compact_plan, layout.anchored_plan, layout.references_plan, layout.ingredients_plan):
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, list):
            continue
        for scene in value:
            if not isinstance(scene, dict):
                continue
            number = int(scene.get("scene") or 0)
            projection = scene.get("canonical_projection")
            projected_id = str(projection.get("scene_id") or "") if isinstance(projection, dict) else ""
            if projected_id and projected_id != expected.get(number, ""):
                findings.append(
                    f"{path.relative_to(project).as_posix()}: scene {number} has stale projection identity",
                )
    return findings
