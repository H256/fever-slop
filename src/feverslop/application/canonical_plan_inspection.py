from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from feverslop.domain.canonical_render_plan import resolve_effective_role, validate_canonical_plan


@dataclass(frozen=True)
class RoleInspection:
    role: str
    generated: Any
    generated_provenance: Mapping[str, Any]
    override: Any | None
    override_provenance: Mapping[str, Any] | None
    effective: Any
    owner: str


def inspect_scene_roles(
    scenes: Sequence[Mapping[str, Any]], scene_number: int,
) -> tuple[str, tuple[RoleInspection, ...]]:
    validate_canonical_plan(list(scenes))
    matches = [scene for scene in scenes if int(scene.get("scene") or 0) == scene_number]
    if len(matches) != 1:
        raise ValueError(f"Canonical scene {scene_number} was not found uniquely")
    canonical = matches[0].get("canonical")
    if not isinstance(canonical, Mapping):
        raise ValueError(f"Scene {scene_number} has no canonical identity")
    roles = canonical.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError(f"Scene {scene_number} canonical roles are invalid")
    inspected = []
    for role_name in sorted(str(key) for key in roles):
        role = roles[role_name]
        if not isinstance(role, Mapping):
            raise ValueError(f"Canonical role {role_name} is invalid")
        generated = role.get("generated")
        override = role.get("override")
        generated_value = generated.get("value") if isinstance(generated, Mapping) else None
        override_value = override.get("value") if isinstance(override, Mapping) else None
        inspected.append(RoleInspection(
            role_name,
            generated_value,
            generated.get("provenance") or {} if isinstance(generated, Mapping) else {},
            override_value,
            override.get("provenance") or {} if isinstance(override, Mapping) else None,
            resolve_effective_role(matches[0], role_name),
            "override" if isinstance(override, Mapping) else "generated",
        ))
    return str(canonical["scene_id"]), tuple(inspected)


def inspect_overrides(scenes: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    validate_canonical_plan(list(scenes))
    result = []
    for scene in scenes:
        canonical = scene.get("canonical")
        if not isinstance(canonical, Mapping):
            continue
        roles = canonical.get("roles") or {}
        for role_name in sorted(roles):
            role = roles[role_name]
            override = role.get("override") if isinstance(role, Mapping) else None
            if isinstance(override, Mapping):
                result.append({
                    "scene": int(scene.get("scene") or 0),
                    "scene_id": str(canonical["scene_id"]),
                    "role": str(role_name),
                    "provenance": dict(override.get("provenance") or {}),
                })
    return tuple(result)
