from __future__ import annotations

from copy import deepcopy
from enum import StrEnum
from typing import Any, Mapping
from uuid import NAMESPACE_URL, uuid5

from feverslop.errors import FeverSlopDataError

CANONICAL_SCHEMA = "feverslop.render-scene.v1"
_MISSING = object()


class PromptRole(StrEnum):
    Z_IMAGE = "z_image.prompt"
    LTX_BASE = "ltx.base"
    LTX_I2V = "ltx.i2v"
    LTX_STATIC = "ltx.static"
    LTX_RELAY = "ltx.relay"
    LTX_MSR_GLOBAL = "ltx.msr.global"
    LTX_MSR_RELAY = "ltx.msr.relay"
    INGREDIENTS_GLOBAL = "ingredients.global"
    INGREDIENTS_RELAY = "ingredients.relay"
    H3_VIDEO = "h3.video"
    PERFORMANCE_TIMING = "performance.timing"


def stable_scene_id(segment_id: str) -> str:
    normalized = str(segment_id).strip()
    if not normalized:
        raise FeverSlopDataError("canonical.segment_id must not be empty")
    return str(uuid5(NAMESPACE_URL, f"feverslop.render-scene:{normalized}"))


def build_canonical_scene(
    *,
    segment_id: str,
    generated_roles: Mapping[str, Any],
    scene_id: str | None = None,
    provenance_source: str = "render-plan-builder",
) -> dict[str, Any]:
    normalized_segment_id = str(segment_id).strip()
    if not normalized_segment_id:
        raise FeverSlopDataError("canonical.segment_id must not be empty")
    normalized_scene_id = str(scene_id or stable_scene_id(normalized_segment_id)).strip()
    if not normalized_scene_id:
        raise FeverSlopDataError("canonical.scene_id must not be empty")

    roles = {
        str(role): {
            "generated": {
                "value": deepcopy(value),
                "provenance": {"source": provenance_source},
            },
        }
        for role, value in generated_roles.items()
    }
    return {
        "schema": CANONICAL_SCHEMA,
        "scene_id": normalized_scene_id,
        "segment_id": normalized_segment_id,
        "roles": roles,
    }


def resolve_effective_role(
    scene: Mapping[str, Any],
    role: str,
    *,
    legacy_value: Any = _MISSING,
    allow_empty: bool = False,
) -> Any:
    role_name = str(role)
    canonical = scene.get("canonical")
    if canonical is None:
        return _resolve_legacy(role_name, legacy_value, allow_empty=allow_empty)
    canonical_payload = _canonical_payload(canonical)
    roles = canonical_payload["roles"]
    role_payload = roles.get(role_name)
    if role_payload is None:
        return _resolve_legacy(role_name, legacy_value, allow_empty=allow_empty)
    role_path = f"canonical.roles.{role_name}"
    if not isinstance(role_payload, Mapping):
        raise FeverSlopDataError(f"{role_path} must be an object")
    if "effective" in role_payload:
        raise FeverSlopDataError(
            f"{role_path}.effective must not be persisted; it is computed at runtime",
        )

    for owner in ("override", "generated"):
        if owner not in role_payload:
            continue
        owned_value = role_payload[owner]
        owner_path = f"{role_path}.{owner}"
        if not isinstance(owned_value, Mapping):
            raise FeverSlopDataError(f"{owner_path} must be an object")
        if "value" not in owned_value:
            raise FeverSlopDataError(f"{owner_path} is missing required key: 'value'")
        value = owned_value["value"]
        _validate_value(value, f"{owner_path}.value", allow_empty=allow_empty)
        return deepcopy(value)

    return _resolve_legacy(role_name, legacy_value, allow_empty=allow_empty)


def validate_canonical_plan(scenes: list[Mapping[str, Any]]) -> None:
    seen_scene_ids: set[str] = set()
    seen_segment_ids: set[str] = set()
    for index, scene in enumerate(scenes):
        canonical = scene.get("canonical")
        if canonical is None:
            continue
        payload = _canonical_payload(canonical, path=f"scenes[{index}].canonical")
        scene_id = payload["scene_id"]
        segment_id = payload["segment_id"]
        if scene_id in seen_scene_ids:
            raise FeverSlopDataError(f"duplicate canonical scene_id: {scene_id}")
        if segment_id in seen_segment_ids:
            raise FeverSlopDataError(f"duplicate canonical segment_id: {segment_id}")
        seen_scene_ids.add(scene_id)
        seen_segment_ids.add(segment_id)


def _canonical_payload(value: Any, *, path: str = "canonical") -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FeverSlopDataError(f"{path} must be an object")
    if value.get("schema") != CANONICAL_SCHEMA:
        raise FeverSlopDataError(
            f"{path}.schema must be {CANONICAL_SCHEMA!r}",
        )
    for key in ("scene_id", "segment_id"):
        field = value.get(key)
        if not isinstance(field, str) or not field.strip():
            raise FeverSlopDataError(f"{path}.{key} must be a non-empty string")
    roles = value.get("roles")
    if not isinstance(roles, Mapping):
        raise FeverSlopDataError(f"{path}.roles must be an object")
    return value


def _resolve_legacy(role: str, value: Any, *, allow_empty: bool) -> Any:
    if value is _MISSING:
        raise FeverSlopDataError(f"canonical role {role!r} has no effective value")
    _validate_value(value, f"legacy value for {role}", allow_empty=allow_empty)
    return deepcopy(value)


def _validate_value(value: Any, path: str, *, allow_empty: bool) -> None:
    if allow_empty:
        return
    if value is None:
        raise FeverSlopDataError(f"{path} must not be empty")
    if isinstance(value, str) and not value.strip():
        raise FeverSlopDataError(f"{path} must not be empty")
    if isinstance(value, (list, tuple, dict)) and not value:
        raise FeverSlopDataError(f"{path} must not be empty")
