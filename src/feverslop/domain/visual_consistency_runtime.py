from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from feverslop.domain.artifact_hash import is_sha256_hex
from feverslop.domain.visual_consistency import SceneConsistencyContract

CONTINUITY_ANCHOR_HEADER = "Continuity anchors (keep unchanged):"


def reference_look_id(scene: dict, *, kind: str, semantic_id: str) -> str:
    look_ids = scene.get("look_ids") or {}
    if kind == "actor":
        values = look_ids.get("actors") if isinstance(look_ids, dict) else {}
        value = values.get(semantic_id) if isinstance(values, dict) else None
        if not value:
            legacy = scene.get("actor_look_ids") or {}
            value = legacy.get(semantic_id) if isinstance(legacy, dict) else None
    else:
        value = look_ids.get("location") if isinstance(look_ids, dict) else None
        value = value or scene.get("location_look_id")
    return str(value or "default").strip() or "default"


def resolve_reference_look(item: dict, look_id: str) -> dict:
    resolved = dict(item)
    if look_id != "default":
        matched = False
        for look in item.get("looks") or []:
            if isinstance(look, dict) and str(look.get("id") or "") == look_id:
                resolved.update(look)
                matched = True
                break
        if not matched:
            raise ValueError(
                f"Reference {item.get('id')!r} has no look {look_id!r}",
            )
    resolved["id"] = item.get("id")
    resolved["name"] = item.get("name")
    resolved["look_id"] = look_id
    return resolved


def ingredients_sheet_signature(
    references: list[dict[str, str]],
    *,
    size: tuple[int, int],
    layout_version: str,
) -> str:
    payload = {
        "layout_version": str(layout_version),
        "size": [int(size[0]), int(size[1])],
        "references": [
            {
                "id": str(reference["id"]),
                "type": str(reference["type"]),
                "sha256": str(reference["sha256"]),
            }
            for reference in references
        ],
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def bind_continuity_anchors(
    prompt: str,
    contract: SceneConsistencyContract | dict | None,
    *,
    max_chars: int = 700,
) -> str:
    text = scrub_prior_context(str(prompt or ""))
    if contract is None:
        return text
    if isinstance(contract, dict):
        contract = SceneConsistencyContract.from_dict(contract)
    text = re.sub(
        rf"\n?{re.escape(CONTINUITY_ANCHOR_HEADER)}[^\n]*",
        "",
        text,
    ).strip()
    anchors = scrub_prior_context(
        contract.prompt_anchor_text(max_chars=max_chars),
    ).replace("\n", "; ")
    return (
        f"{text}\n{CONTINUITY_ANCHOR_HEADER} {anchors}".strip()
        if anchors
        else text
    )


def validate_runtime_visual_consistency(
    scene: dict,
    *,
    mode: str,
    workflow_profile: str,
) -> SceneConsistencyContract | None:
    payload = scene.get("visual_consistency")
    if payload is None:
        return None
    try:
        contract = SceneConsistencyContract.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid visual consistency contract: {exc}") from exc
    if contract.scene != scene.get("scene"):
        raise ValueError("visual consistency scene does not match runtime scene")
    if contract.mode != mode:
        raise ValueError(
            f"visual consistency mode {contract.mode!r} does not match backend {mode!r}",
        )
    if contract.workflow_profile != workflow_profile:
        raise ValueError(
            "visual consistency workflow profile "
            f"{contract.workflow_profile!r} does not match backend "
            f"{workflow_profile!r}",
        )
    _validate_reference_bindings(scene, contract)
    if mode == "ingredients":
        _validate_ingredients_signature(scene, contract)
    elif mode == "msr":
        _validate_msr_bindings(scene, contract)
    return contract


def _validate_reference_bindings(
    scene: dict,
    contract: SceneConsistencyContract,
) -> None:
    references = scene.get("references") or {}
    actor_ids = references.get("actor_ids")
    if actor_ids is None:
        actor_ids = (scene.get("reference_ids") or {}).get("actors") or []
    location_id = references.get("location_id")
    if location_id is None:
        location_id = (scene.get("reference_ids") or {}).get("location") or ""
    if [anchor.id for anchor in contract.actors] != list(actor_ids or []):
        raise ValueError("visual consistency actor references do not match runtime scene")
    expected_location = contract.location.id if contract.location is not None else ""
    if expected_location != str(location_id or ""):
        raise ValueError("visual consistency location reference does not match runtime scene")
    if any(
        anchor.asset_role != "identity-reference"
        for anchor in contract.actors
    ):
        raise ValueError("visual consistency actor reference role mismatch")
    if (
        contract.location is not None
        and contract.location.asset_role != "environment-reference"
    ):
        raise ValueError("visual consistency location reference role mismatch")


def _validate_ingredients_signature(
    scene: dict,
    contract: SceneConsistencyContract,
) -> None:
    ingredients = scene.get("ingredients") or {}
    signature = str(ingredients.get("signature") or "").strip()
    layout_version = str(ingredients.get("layout_version") or "").strip()
    size = ingredients.get("size") or []
    references = ingredients.get("signature_references") or []
    sources = ingredients.get("signature_sources") or []
    expected_bindings = [
        {"id": anchor.id, "type": anchor.kind}
        for anchor in (
            *contract.actors,
            *(() if contract.location is None else (contract.location,)),
        )
    ]
    actual_bindings = [
        {"id": reference.get("id"), "type": reference.get("type")}
        for reference in references
        if isinstance(reference, dict)
    ]
    if actual_bindings != expected_bindings:
        raise ValueError("Ingredients signature reference roles do not match contract")
    source_bindings = [
        {"id": source.get("id"), "type": source.get("type")}
        for source in sources
        if isinstance(source, dict)
    ]
    if source_bindings != actual_bindings:
        raise ValueError("Ingredients signature source roles do not match signature")
    if any(
        not isinstance(source.get("path"), str) or not source["path"].strip()
        for source in sources
        if isinstance(source, dict)
    ):
        raise ValueError("Ingredients signature source path is missing")
    for reference in references:
        value = reference.get("sha256") if isinstance(reference, dict) else None
        if not is_sha256_hex(value):
            raise ValueError("Ingredients signature reference hash is invalid")
    if len(size) != 2:
        raise ValueError("Ingredients signature size is missing")
    expected = ingredients_sheet_signature(
        references,
        size=(int(size[0]), int(size[1])),
        layout_version=layout_version,
    )
    if signature != expected:
        raise ValueError("Ingredients signature does not match reference contract")
    sheet_path = Path(str(ingredients.get("sheet_path") or ""))
    if sheet_path.stem != signature or sheet_path.parent.name != "by_signature":
        raise ValueError("Ingredients sheet path does not match signature")


def _validate_msr_bindings(
    scene: dict,
    contract: SceneConsistencyContract,
) -> None:
    references = scene.get("references") or {}
    actor_paths = (
        references.get("actor_msr_paths")
        or references.get("actor_sheet_paths")
        or []
    )
    if len(actor_paths) != len(contract.actors):
        raise ValueError("MSR actor reference roles do not match contract")
    location_path = (
        references.get("location_msr_path")
        or references.get("location_sheet_path")
    )
    if bool(location_path) != (contract.location is not None):
        raise ValueError("MSR location reference role does not match contract")


def scrub_prior_context(value: str) -> str:
    text = str(value or "").strip()
    patterns = (
        (
            r"continue\s+with\s+(?:the\s+)?same\b"
            r"[^.;!?\n]*?\bfrom\s+before\b"
            r"\s*(?:[,;:.!?-]\s*|$)"
        ),
        (
            r"(?:same\s+as\s+before|as\s+before)\b"
            r"(?=\s+(?:after|from|in|during)\b)[^.;!?\n]*?"
            r"(?:[,;.!?]\s*|$)"
        ),
        r"(?:same\s+as\s+before|as\s+before)\b\s*[,;:.-]?\s*",
        (
            r"(?:(?:after|from|in|during)\s+(?:the\s+)?)?"
            r"(?:prior|previous)\s+(?:scene|shot)(?:\s+aside)?"
            r"(?:\s+and\s+(?:the\s+)?(?:prior|previous)\s+"
            r"(?:scene|shot)(?:\s+aside)?)*\s*[,;:.-]\s*"
        ),
    )
    while text:
        updated = text
        for pattern in patterns:
            updated = re.sub(
                rf"(^|[\n;:.!?]\s*){pattern}",
                r"\1",
                updated,
                count=1,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            if updated != text:
                break
        if updated == text:
            break
        text = updated.lstrip()
    text = re.sub(r"^(?:and|or)\s+", "", text, flags=re.IGNORECASE)
    return re.sub(r"[ \t]+", " ", text).strip()
