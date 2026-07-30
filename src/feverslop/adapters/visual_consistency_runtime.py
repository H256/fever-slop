from __future__ import annotations

from pathlib import Path

from PIL import Image

from feverslop.domain.prepared_workflow import sha256_file
from feverslop.domain.visual_consistency_runtime import (
    validate_runtime_visual_consistency as validate_runtime_metadata,
)


def validate_backend_visual_consistency(
    scene: dict,
    *,
    mode: str,
    workflow_profile: str,
    project_dir: Path | None,
) -> None:
    contract = validate_runtime_metadata(
        scene,
        mode=mode,
        workflow_profile=workflow_profile,
    )
    if contract is None:
        return
    if project_dir is None:
        raise ValueError(
            "visual consistency source verification requires a project directory"
        )
    root = Path(project_dir).resolve()
    if mode == "msr":
        _reject_absolute_msr_paths(scene, contract)
    _validate_contract_sources(scene, contract, root=root)
    if mode == "ingredients":
        ingredients = scene.get("ingredients") or {}
        references = ingredients.get("signature_references") or []
        sources = ingredients.get("signature_sources") or []
        for reference, source in zip(references, sources):
            if Path(source["path"]).is_absolute():
                raise ValueError(
                    "Ingredients signature source path must be project-relative"
                )
            actual = sha256_file(_contained_source(root, source["path"]))
            if actual != reference["sha256"]:
                raise ValueError(
                    f"Ingredients source hash mismatch for {reference['id']!r}"
                )
        _validate_ingredients_sheet(ingredients, root=root)
    elif mode == "msr":
        references = scene.get("references") or {}
        actor_paths = (
            references.get("actor_msr_paths")
            or references.get("actor_sheet_paths")
            or []
        )
        for path, anchor in zip(actor_paths, contract.actors):
            source = _contained_source(
                root,
                path,
                reference_kind="actor",
                reference_id=anchor.id,
            )
            if sha256_file(source) != anchor.asset_sha256:
                raise ValueError(
                    f"MSR actor reference hash mismatch for {anchor.id!r}"
                )
        location_path = (
            references.get("location_msr_path")
            or references.get("location_sheet_path")
        )
        if contract.location is not None and (
            sha256_file(
                _contained_source(
                    root,
                    location_path,
                    reference_kind="location",
                    reference_id=contract.location.id,
                )
            )
            != contract.location.asset_sha256
        ):
            raise ValueError(
                f"MSR location reference hash mismatch for "
                f"{contract.location.id!r}"
            )


def _reject_absolute_msr_paths(scene: dict, contract) -> None:
    references = scene.get("references") or {}
    actor_paths = (
        references.get("actor_msr_paths")
        or references.get("actor_sheet_paths")
        or []
    )
    for path, anchor in zip(actor_paths, contract.actors):
        if Path(path).is_absolute():
            raise ValueError(
                f"actor reference {anchor.id!r} path must be project-relative: "
                f"{path}"
            )
    location_path = (
        references.get("location_msr_path")
        or references.get("location_sheet_path")
    )
    if (
        contract.location is not None
        and location_path
        and Path(location_path).is_absolute()
    ):
        raise ValueError(
            f"location reference {contract.location.id!r} path must be "
            f"project-relative: {location_path}"
        )


def _validate_contract_sources(scene: dict, contract, *, root: Path) -> None:
    metadata = scene.get("visual_consistency_sources") or {}
    actor_sources = metadata.get("actors") or []
    if [item.get("id") for item in actor_sources] != [
        anchor.id for anchor in contract.actors
    ]:
        raise ValueError("visual consistency actor source bindings do not match contract")
    for item, anchor in zip(actor_sources, contract.actors):
        source = _contained_source(
            root,
            item.get("path"),
            reference_kind="actor",
            reference_id=anchor.id,
        )
        if sha256_file(source) != anchor.asset_sha256:
            raise ValueError(
                f"visual consistency contract source hash mismatch for actor "
                f"{anchor.id!r}"
            )
    location_source = metadata.get("location")
    if bool(location_source) != (contract.location is not None):
        raise ValueError(
            "visual consistency location source binding does not match contract"
        )
    if contract.location is not None:
        if location_source.get("id") != contract.location.id:
            raise ValueError(
                "visual consistency location source binding does not match contract"
            )
        source = _contained_source(
            root,
            location_source.get("path"),
            reference_kind="location",
            reference_id=contract.location.id,
        )
        if sha256_file(source) != contract.location.asset_sha256:
            raise ValueError(
                f"visual consistency contract source hash mismatch for location "
                f"{contract.location.id!r}"
            )


def _validate_ingredients_sheet(ingredients: dict, *, root: Path) -> None:
    raw = Path(str(ingredients.get("sheet_path") or ""))
    if raw.is_absolute():
        raise ValueError("Ingredients sheet path must be project-relative")
    sheet = (root / raw).resolve()
    allowed_roots = (
        (
            root
            / "output"
            / "references"
            / "ingredients_sheets"
            / "by_signature"
        ).resolve(),
        (
            root
            / "movie"
            / "references"
            / "ingredients_sheets"
            / "by_signature"
        ).resolve(),
    )
    if sheet.parent not in allowed_roots:
        raise ValueError(
            "Ingredients sheet must be in a canonical Ingredients cache"
        )
    if not sheet.is_file():
        raise ValueError("Ingredients sheet is missing or not a file")
    expected_size = tuple(int(value) for value in ingredients.get("size") or [])
    try:
        with Image.open(sheet) as image:
            if image.format != "PNG":
                raise ValueError("Ingredients sheet must be a readable PNG")
            if image.size != expected_size:
                raise ValueError(
                    "Ingredients sheet dimensions do not match contract"
                )
            image.verify()
    except (OSError, SyntaxError) as exc:
        raise ValueError("Ingredients sheet must be a readable PNG") from exc
    expected_hash = str(ingredients.get("sheet_sha256") or "")
    if len(expected_hash) != 64 or sha256_file(sheet) != expected_hash:
        raise ValueError("Ingredients sheet hash does not match runtime metadata")


def _contained_source(
    root: Path,
    value: str | Path | None,
    *,
    reference_kind: str = "source",
    reference_id: str = "",
) -> Path:
    if value is None:
        raise ValueError(
            f"{reference_kind} reference {reference_id!r} source path is missing"
        )
    raw = Path(value)
    if raw.is_absolute():
        raise ValueError(
            f"{reference_kind} reference {reference_id!r} path must be "
            f"project-relative: {raw}"
        )
    resolved = (root / raw).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(
            f"{reference_kind} reference {reference_id!r} path escapes the "
            f"project: {raw}"
        )
    if not resolved.is_file():
        raise ValueError(
            f"{reference_kind} reference {reference_id!r} path is missing or "
            f"not a file: {raw}"
        )
    return resolved
