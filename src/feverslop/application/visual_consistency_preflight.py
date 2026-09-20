from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from feverslop.application.visual_consistency import (
    build_scene_contract,
    normalize_reference_ids,
)
from feverslop.domain.scene_cast import resolve_scene_cast
from feverslop.domain.visual_consistency import (
    ConsistencyIssue,
    PreflightMode,
    SceneConsistencyContract,
    can_handoff,
)
from feverslop.application.visual_consistency import actor_look_id, location_look_id
from feverslop.domain.ensemble import EnsembleConfig, validate_ensemble_cast
from feverslop.domain.semantic_intent import IntentLedger, entity_constraint_ids
from feverslop.ports.visual_consistency import ReferenceManifestSnapshot


def resolve_preflight_workflow_profile(
    scenes: list[dict[str, Any]],
    *,
    explicit_profile: str | None,
    legacy_fallback: str | None,
) -> str:
    """Resolve one profile without silently rewriting stored contracts."""
    stored_profiles: set[str] = set()
    has_structured_contract = False
    for scene in scenes:
        payload = scene.get("visual_consistency")
        if not isinstance(payload, dict):
            continue
        has_structured_contract = True
        profile = payload.get("workflow_profile")
        if isinstance(profile, str) and profile.strip():
            stored_profiles.add(profile.strip())

    if len(stored_profiles) > 1:
        profiles = ", ".join(sorted(stored_profiles))
        raise ValueError(
            f"Render plan contains mixed visual consistency workflow profiles: {profiles}",
        )
    if stored_profiles:
        stored_profile = next(iter(stored_profiles))
        if explicit_profile and explicit_profile != stored_profile:
            raise ValueError(
                "Selected workflow profile does not match the stored visual "
                f"consistency profile: {explicit_profile} != {stored_profile}",
            )
        return stored_profile
    if has_structured_contract:
        raise ValueError(
            "Structured visual consistency contracts must declare workflow_profile",
        )
    if explicit_profile:
        return explicit_profile
    if legacy_fallback:
        return legacy_fallback
    raise ValueError("A visual consistency workflow profile is required")


@dataclass(frozen=True)
class VisualConsistencyPreflightResult:
    contracts: tuple[SceneConsistencyContract, ...]
    issues: tuple[ConsistencyIssue, ...]

    @property
    def renderable(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


def preflight_visual_consistency(
    scenes: Iterable[Mapping[str, Any]],
    snapshot: ReferenceManifestSnapshot,
    *,
    mode: str,
    workflow_profile: str,
    preflight_mode: PreflightMode | str = PreflightMode.WARN,
    subject_mode: str = "multi",
    max_scene_actors: int = 4,
    supports_continuous_transitions: bool = True,
    ensembles: tuple[EnsembleConfig, ...] = (),
    semantic_intent_ledger: IntentLedger | None = None,
) -> VisualConsistencyPreflightResult:
    policy = PreflightMode.parse(preflight_mode)
    if policy is PreflightMode.OFF:
        return VisualConsistencyPreflightResult((), ())

    scene_items = tuple(scenes)
    contracts: list[SceneConsistencyContract] = []
    issues: list[ConsistencyIssue] = []
    seen_scene_numbers: set[int] = set()
    previous_contract: SceneConsistencyContract | None = None
    cast_universe, scene_presence = _semantic_cast_universe(scene_items)
    issues.extend(
        _check_semantic_project_presence(
            semantic_intent_ledger,
            cast_universe,
            scene_presence,
            policy,
        )
    )
    for scene in scene_items:
        scene_number = scene.get("scene")
        if type(scene_number) is not int or scene_number <= 0:
            raise ValueError("scene must be a positive integer")
        if scene_number in seen_scene_numbers:
            issues.append(
                _issue(
                    "duplicate_scene_number",
                    scene_number,
                    f"Scene number {scene_number} appears more than once",
                    policy,
                ),
            )
        seen_scene_numbers.add(scene_number)
        malformed = _malformed_reference_bindings(scene)
        if malformed:
            issues.append(
                _issue(
                    "malformed_reference_bindings",
                    scene_number,
                    f"Scene {scene_number} has malformed reference bindings: "
                    f"{malformed}",
                    policy,
                ),
            )
            previous_contract = None
            continue
        actor_ids, location_id = normalize_reference_ids(scene)
        if not actor_ids and not location_id:
            issues.append(
                _issue(
                    "legacy_contract_unknown",
                    scene_number,
                    f"Scene {scene_number} has no structured reference bindings",
                    policy,
                ),
            )
            previous_contract = None
            continue
        missing_bindings, binding_issues = _check_scene_bindings(
            scene,
            scene_number,
            actor_ids,
            location_id,
            snapshot,
            subject_mode=subject_mode,
            max_scene_actors=max_scene_actors,
            policy=policy,
        )
        issues.extend(binding_issues)
        if missing_bindings:
            previous_contract = None
            continue
        issues.extend(
            _check_ensemble_bindings(
                scene,
                scene_number,
                actor_ids,
                ensembles,
                policy,
            ),
        )
        issues.extend(
            _check_semantic_scene_presence(
                semantic_intent_ledger,
                scene_number,
                actor_ids,
                location_id,
                policy,
            ),
        )
        contract = build_scene_contract(
            scene,
            snapshot,
            mode=mode,
            workflow_profile=workflow_profile,
        )
        contracts.append(contract)
        issues.extend(_mode_issues(scene, contract, actor_ids, location_id, policy))
        if contract.transition_from_previous == "continuous":
            supported = supports_continuous_transitions and (
                previous_contract is not None
                and previous_contract.scene + 1 == contract.scene
                and can_handoff(previous_contract, contract)
            )
            if not supported:
                issues.append(
                    _issue(
                        "unsupported_continuous_transition",
                        scene_number,
                        f"Scene {scene_number} requests a continuous transition "
                        "that the workflow or adjacent contracts cannot support",
                        policy,
                    ),
                )
        stored = scene.get("visual_consistency")
        stored_fingerprint = (
            stored.get("fingerprint") if isinstance(stored, Mapping) else None
        )
        if "visual_consistency" in scene and (
            not isinstance(stored_fingerprint, str)
            or not stored_fingerprint.strip()
            or stored_fingerprint != contract.fingerprint
        ):
            issues.append(
                _issue(
                    "visual_consistency_fingerprint_mismatch",
                    scene_number,
                    f"Scene {scene_number} has a stale visual consistency fingerprint",
                    policy,
                ),
            )
        previous_contract = contract
    return VisualConsistencyPreflightResult(tuple(contracts), tuple(issues))


def _check_scene_bindings(
    scene: Mapping[str, Any],
    scene_number: int,
    actor_ids: tuple[str, ...],
    location_id: str,
    snapshot: ReferenceManifestSnapshot,
    *,
    subject_mode: str,
    max_scene_actors: int,
    policy: PreflightMode,
) -> tuple[bool, list[ConsistencyIssue]]:
    """Check cast limits and reference availability for one scene's bindings."""
    issues: list[ConsistencyIssue] = []
    available_actors = [
        {"id": actor_id, "name": actor_id}
        for actor_id in dict.fromkeys(key[0] for key in snapshot.actors)
    ]
    known_selected = tuple(
        actor_id
        for actor_id in actor_ids
        if any(key[0] == actor_id for key in snapshot.actors)
    )
    cast = resolve_scene_cast(
        selected_actor_ids=known_selected,
        available_actors=available_actors,
        subject_mode=str(
            (scene.get("references") or {}).get("subject_mode")
            or subject_mode
        ),
        max_scene_actors=max_scene_actors,
        scene_number=scene_number,
    )
    if known_selected and cast.visible_actor_ids != known_selected:
        issues.append(
            _issue(
                "subject_limit_exceeded",
                scene_number,
                f"Scene {scene_number} exceeds the configured visible actor limit",
                policy,
            ),
        )
    missing_bindings = False
    for actor_id in actor_ids:
        if (actor_id, _actor_look(scene, actor_id)) in snapshot.actors:
            continue
        missing_bindings = True
        actor_exists = any(key[0] == actor_id for key in snapshot.actors)
        issues.append(
            _issue(
                "missing_actor_look" if actor_exists else "missing_actor_reference",
                scene_number,
                (
                    f"Scene {scene_number} references unavailable look "
                    f"{_actor_look(scene, actor_id)!r} for actor {actor_id!r}"
                    if actor_exists
                    else f"Scene {scene_number} references unavailable actor "
                    f"{actor_id!r}"
                ),
                policy,
            ),
        )
    if location_id and (
        location_id,
        _location_look(scene),
    ) not in snapshot.locations:
        missing_bindings = True
        location_exists = any(
            key[0] == location_id for key in snapshot.locations
        )
        issues.append(
            _issue(
                (
                    "missing_location_look"
                    if location_exists
                    else "missing_location_reference"
                ),
                scene_number,
                (
                    f"Scene {scene_number} references unavailable look "
                    f"{_location_look(scene)!r} for location {location_id!r}"
                    if location_exists
                    else f"Scene {scene_number} references unavailable location "
                    f"{location_id!r}"
                ),
                policy,
            ),
        )
    return missing_bindings, issues


def _mode_issues(
    scene: Mapping[str, Any],
    contract: SceneConsistencyContract,
    actor_ids: tuple[str, ...],
    location_id: str,
    policy: PreflightMode,
) -> list[ConsistencyIssue]:
    issues: list[ConsistencyIssue] = []
    references = scene.get("references")
    references = references if isinstance(references, Mapping) else {}
    if contract.mode == "ingredients":
        ingredients = scene.get("ingredients")
        ingredients = ingredients if isinstance(ingredients, Mapping) else {}
        sheet = ingredients.get("sheet_path") or scene.get(
            "ingredients_scene_sheet",
        )
        if not isinstance(sheet, str) or not sheet.strip():
            issues.append(
                _issue(
                    "missing_ingredients_sheet",
                    contract.scene,
                    f"Scene {contract.scene} has no Ingredients sheet",
                    policy,
                ),
            )
        anchors = ingredients.get("anchors") or scene.get(
            "ingredients_scene_sheet_anchors",
        )
        anchor_ids = {
            str(anchor.get("id") or "").strip()
            for anchor in anchors or ()
            if isinstance(anchor, Mapping)
        }
        expected_ids = {*actor_ids, *(() if not location_id else (location_id,))}
        if anchor_ids != expected_ids:
            issues.append(
                _issue(
                    "missing_ingredients_anchor",
                    contract.scene,
                    f"Scene {contract.scene} Ingredients anchors do not match references",
                    policy,
                ),
            )
    elif contract.mode == "msr":
        raw_actor_roles = (
            references.get("actor_msr_paths")
            or references.get("actor_sheet_paths")
            or ()
        )
        actor_roles = (
            tuple(
                value.strip()
                for value in raw_actor_roles
                if isinstance(value, str) and value.strip()
            )
            if isinstance(raw_actor_roles, (list, tuple))
            else ()
        )
        if actor_ids and len(actor_roles) < len(actor_ids):
            issues.append(
                _issue(
                    "missing_msr_actor_role",
                    contract.scene,
                    f"Scene {contract.scene} has no MSR role for every actor",
                    policy,
                ),
            )
        raw_location_role = (
            references.get("location_msr_path")
            or references.get("location_sheet_path")
        )
        location_role = (
            raw_location_role.strip()
            if isinstance(raw_location_role, str)
            else ""
        )
        if location_id and not location_role:
            issues.append(
                _issue(
                    "missing_msr_location_role",
                    contract.scene,
                    f"Scene {contract.scene} has no MSR location role",
                    policy,
                ),
            )
    return issues


def _check_ensemble_bindings(
    scene: Mapping[str, Any],
    scene_number: int,
    actor_ids: tuple[str, ...],
    ensembles: tuple[EnsembleConfig, ...],
    policy: PreflightMode,
) -> list[ConsistencyIssue]:
    """Enforce all-members-together for ensembles; never auto-add members."""
    issues: list[ConsistencyIssue] = []
    scene_type = _scene_type(scene)
    for ensemble in ensembles:
        missing = validate_ensemble_cast(
            ensemble,
            scene_actor_ids=actor_ids,
            scene_type=scene_type,
        )
        if not missing:
            continue
        issues.append(
            _issue(
                "ensemble_incomplete",
                scene_number,
                (
                    f"Scene {scene_number} is a {scene_type or 'required'} scene "
                    f"for ensemble {ensemble.id!r} but is missing members "
                    f"{', '.join(missing)}; missing members are not added "
                    "automatically"
                ),
                policy,
            ),
        )
    return issues


def _semantic_cast_universe(
    scene_items: tuple[Mapping[str, Any], ...],
) -> tuple[set[str], dict[int, set[str]]]:
    """Collect the union of every scene's actor/location ids and per-scene presence.

    The cast universe is the id namespace the ledger obligations are evaluated
    against. Guarding by it keeps the gate kind-agnostic and prevents false
    positives when a ledger entity id is not referenced by any scene.
    """
    universe: set[str] = set()
    presence: dict[int, set[str]] = {}
    for scene in scene_items:
        scene_number = scene.get("scene")
        if type(scene_number) is not int or scene_number <= 0:
            continue
        actor_ids, location_id = normalize_reference_ids(scene)
        present = set(actor_ids)
        if location_id:
            present.add(location_id)
        universe.update(present)
        presence[scene_number] = present
    return universe, presence


def _check_semantic_project_presence(
    ledger: IntentLedger | None,
    cast_universe: set[str],
    scene_presence: dict[int, set[str]],
    policy: PreflightMode,
) -> list[ConsistencyIssue]:
    """Fail required entities that never appear in any scene.

    A ``required`` entity whose recurrence is ``once`` or ``recurring`` must be
    present in at least one scene. ``always`` entities are checked per scene.
    No issues when the ledger is None (legacy/no declared intent) or empty.
    """
    if ledger is None:
        return []
    issues: list[ConsistencyIssue] = []
    for entity in ledger.entities:
        if entity.status != "required" or entity.recurrence == "always":
            continue
        if entity.id in cast_universe:
            continue
        issues.append(
            _issue(
                "semantic_required_missing",
                next(iter(scene_presence)) if scene_presence else 1,
                (
                    f"Required entity {entity.id!r} (constraint ids "
                    f"{_constraint_ids_text(ledger, entity.id)}) is not present "
                    "in any scene; it was not added automatically"
                ),
                policy,
            ),
        )
    return issues


def _check_semantic_scene_presence(
    ledger: IntentLedger | None,
    scene_number: int,
    actor_ids: tuple[str, ...],
    location_id: str,
    policy: PreflightMode,
) -> list[ConsistencyIssue]:
    """Enforce per-scene always-presence and required co-presence.

    ``required`` + ``always`` entities must be present in every scene.
    Required relations enforce co-presence: when the subject is present, the
    target must be too. Both are evaluated per scene against that scene's
    actual presence, so a ledger id absent from a given scene is flagged only
    for that scene. No issues when the ledger is None (legacy/no declared
    intent) or empty.
    """
    if ledger is None:
        return []
    present = set(actor_ids)
    if location_id:
        present.add(location_id)
    issues: list[ConsistencyIssue] = []
    for entity in ledger.entities:
        if entity.status != "required" or entity.recurrence != "always":
            continue
        if entity.id in present:
            continue
        issues.append(
            _issue(
                "semantic_always_missing",
                scene_number,
                (
                    f"Scene {scene_number} is missing always-present required "
                    f"entity {entity.id!r} (constraint ids "
                    f"{_constraint_ids_text(ledger, entity.id)})"
                ),
                policy,
            ),
        )
    for relation in ledger.relations:
        if relation.status != "required":
            continue
        if relation.subject_id not in present:
            continue
        if relation.target_id in present:
            continue
        issues.append(
            _issue(
                "semantic_co_presence",
                scene_number,
                (
                    f"Scene {scene_number} has {relation.subject_id!r} but not "
                    f"co-present required target {relation.target_id!r} "
                    f"(relation {relation.id!r})"
                ),
                policy,
            ),
        )
    return issues


def _constraint_ids_text(ledger: IntentLedger, entity_id: str) -> str:
    ids = entity_constraint_ids(ledger, entity_id)
    return ", ".join(ids) if ids else "none"


def _scene_type(scene: Mapping[str, Any]) -> str:
    metadata = scene.get("metadata")
    if isinstance(metadata, Mapping):
        raw = metadata.get("type")
        if isinstance(raw, str) and raw.strip():
            return raw.strip().lower()
    raw = scene.get("type")
    if isinstance(raw, str) and raw.strip():
        return raw.strip().lower()
    return ""


def _malformed_reference_bindings(scene: Mapping[str, Any]) -> str:
    candidates = (
        ("reference_ids", scene.get("reference_ids"), "actors", "location"),
        ("references", scene.get("references"), "actor_ids", "location_id"),
        ("scene", scene, "actor_ids", "location_id"),
    )
    for container_name, raw_values, actor_field, _location_field in candidates:
        if raw_values is None:
            continue
        if not isinstance(raw_values, Mapping):
            return f"{container_name} must be an object"
        if actor_field not in raw_values:
            continue
        actors = raw_values[actor_field]
        if not isinstance(actors, (list, tuple)) or any(
            not isinstance(actor, str) or not actor.strip()
            for actor in actors
        ):
            return (
                f"{container_name}.{actor_field} must be a sequence "
                "of nonblank strings"
            )
        if actors:
            break
    for container_name, raw_values, _actor_field, location_field in candidates:
        if raw_values is None:
            continue
        if not isinstance(raw_values, Mapping):
            return f"{container_name} must be an object"
        if location_field not in raw_values:
            continue
        location = raw_values[location_field]
        if not isinstance(location, str):
            return f"{container_name}.{location_field} must be a string"
        if location.strip():
            break
    return ""


def _issue(
    code: str,
    scene: int,
    message: str,
    mode: PreflightMode,
) -> ConsistencyIssue:
    return ConsistencyIssue(
        code=code,
        scene=scene,
        severity="error" if mode is PreflightMode.STRICT else "warning",
        message=message,
    )


def _actor_look(scene: Mapping[str, Any], actor_id: str) -> str:
    return actor_look_id(scene, actor_id)


def _location_look(scene: Mapping[str, Any]) -> str:
    return location_look_id(scene)
