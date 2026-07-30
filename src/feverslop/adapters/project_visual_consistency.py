from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from feverslop.domain.prepared_workflow import sha256_file
from feverslop.domain.visual_consistency import (
    ConsistencyIssue,
    PreflightMode,
    ReferenceAnchor,
)
from feverslop.path_utils import coerce_local_path
from feverslop.ports.visual_consistency import ReferenceManifestSnapshot


class ProjectReferenceManifestAdapter:
    def __init__(self, project_root: Callable[[str], Path]) -> None:
        self._project_root = project_root

    def load(self, project_id: str) -> ReferenceManifestSnapshot:
        root = Path(self._project_root(project_id)).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Project not found: {project_id}")

        actors: dict[tuple[str, str], ReferenceAnchor] = {}
        locations: dict[tuple[str, str], ReferenceAnchor] = {}
        actor_sources: dict[tuple[str, str], str] = {}
        location_sources: dict[tuple[str, str], str] = {}
        decoded_manifests: list[dict[str, Any]] = []

        references_root = root / "output" / "references"
        for kind, destination, sources in (
            ("actor", actors, actor_sources),
            ("location", locations, location_sources),
        ):
            manifest_root = references_root / f"{kind}s"
            for path in sorted(manifest_root.glob("*/manifest.json")):
                decoded = self._read_manifest(root, path)
                source_path = path.relative_to(root).as_posix()
                decoded_manifests.append(
                    {"path": source_path, "manifest": decoded}
                )
                self._add_item(
                    root,
                    destination,
                    sources,
                    kind,
                    decoded,
                    path,
                    source_path,
                )

        movie_path = root / "movie" / "references" / "manifest.json"
        if movie_path.exists():
            decoded = self._read_manifest(root, movie_path)
            source_path = movie_path.relative_to(root).as_posix()
            decoded_manifests.append(
                {
                    "path": source_path,
                    "manifest": decoded,
                }
            )
            for kind, collection_name, destination, sources in (
                ("actor", "actors", actors, actor_sources),
                ("location", "locations", locations, location_sources),
            ):
                collection = decoded.get(collection_name)
                if collection is None:
                    collection = []
                if not isinstance(collection, list):
                    raise ValueError(
                        f"Reference manifest {collection_name} must be a JSON array: "
                        f"{movie_path}"
                    )
                for index, item in enumerate(collection):
                    if not isinstance(item, dict):
                        raise ValueError(
                            f"Reference manifest {collection_name}[{index}] "
                            f"must be a JSON object: {movie_path}"
                        )
                    self._add_item(
                        root,
                        destination,
                        sources,
                        kind,
                        item,
                        movie_path,
                        source_path,
                        item_context=f"{collection_name}[{index}]",
                    )

        canonical = json.dumps(
            decoded_manifests,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return ReferenceManifestSnapshot(
            actors=actors,
            locations=locations,
            revision=hashlib.sha256(canonical).hexdigest(),
        )

    @staticmethod
    def _read_manifest(root: Path, path: Path) -> dict[str, Any]:
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Reference manifest not found: {path}") from None
        if not resolved.is_relative_to(root):
            raise ValueError(f"Reference manifest is outside project root: {path}")
        try:
            decoded = json.loads(resolved.read_text(encoding="utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed reference manifest JSON: {path}") from exc
        if not isinstance(decoded, dict):
            raise ValueError(f"Reference manifest must be a JSON object: {path}")
        return decoded

    @classmethod
    def _add_item(
        cls,
        root: Path,
        destination: dict[tuple[str, str], ReferenceAnchor],
        sources: dict[tuple[str, str], str],
        kind: str,
        item: Mapping[str, Any],
        manifest_path: Path,
        source_path: str,
        *,
        item_context: str = "",
    ) -> None:
        context = (
            f"{source_path} {item_context}" if item_context else source_path
        )
        semantic_id = cls._required_string(item.get("id"), "id", context)
        if not semantic_id:
            raise ValueError(f"Reference id is required: {manifest_path}")

        default_path = cls._asset_value(item, kind, context)
        if default_path:
            cls._insert_anchor(
                destination,
                sources,
                cls._make_anchor(
                    root,
                    kind=kind,
                    semantic_id=semantic_id,
                    look_id="default",
                    asset_value=default_path,
                    description=cls._description(item, context),
                    manifest_path=manifest_path,
                ),
                source_path,
            )

        looks = item.get("looks")
        if looks is None:
            looks = []
        if not isinstance(looks, list):
            raise ValueError(f"Reference looks must be a JSON array: {manifest_path}")
        for index, look in enumerate(looks):
            if not isinstance(look, dict):
                raise ValueError(
                    f"Reference look at index {index} must be a JSON object: "
                    f"{manifest_path}"
                )
            look_context = f"{context} looks[{index}]"
            look_id = cls._required_string(look.get("id"), "look id", look_context)
            asset_value = cls._required_string(
                look.get("sheet_path"),
                "look asset path",
                look_context,
            )
            if not asset_value:
                raise ValueError(
                    f"Reference look sheet_path is required for {semantic_id!r} "
                    f"look {look_id!r}: {manifest_path}"
                )
            cls._insert_anchor(
                destination,
                sources,
                cls._make_anchor(
                    root,
                    kind=kind,
                    semantic_id=semantic_id,
                    look_id=look_id,
                    asset_value=asset_value,
                    description=cls._required_string(
                        look.get("visual_description"),
                        "look visual description",
                        look_context,
                    ),
                    manifest_path=manifest_path,
                ),
                source_path,
            )

    @classmethod
    def _asset_value(
        cls,
        item: Mapping[str, Any],
        kind: str,
        context: str,
    ) -> str:
        equivalent = "msr_input_path" if kind == "actor" else "msr_background_path"
        return cls._first_string(
            item,
            ("msr_sheet_path", equivalent, "sheet_path"),
            "asset path",
            context,
        )

    @classmethod
    def _description(cls, item: Mapping[str, Any], context: str) -> str:
        return cls._first_string(
            item,
            ("visual_description", "image_prompt", "prompt"),
            "visual description",
            context,
        )

    @staticmethod
    def _required_string(value: Any, field: str, context: str) -> str:
        if not isinstance(value, str):
            raise ValueError(
                f"Reference {field} must be a string: {context}"
            )
        return value.strip()

    @classmethod
    def _first_string(
        cls,
        item: Mapping[str, Any],
        keys: tuple[str, ...],
        field: str,
        context: str,
    ) -> str:
        values: list[str] = []
        for key in keys:
            if key not in item or item[key] is None:
                values.append("")
                continue
            values.append(cls._required_string(item[key], field, context))
        return next((value for value in values if value), "")

    @classmethod
    def _make_anchor(
        cls,
        root: Path,
        *,
        kind: str,
        semantic_id: str,
        look_id: str,
        asset_value: str,
        description: str,
        manifest_path: Path,
    ) -> ReferenceAnchor:
        normalized_description = " ".join(description.split())
        if not normalized_description:
            raise ValueError(
                f"Reference visual description is required for {kind} "
                f"{semantic_id!r} look {look_id!r}: {manifest_path}"
            )
        asset = cls._resolve_asset(root, asset_value)
        prefix = f"Reference {kind} `{semantic_id}` (look `{look_id}`): "
        return ReferenceAnchor(
            id=semantic_id,
            kind=kind,
            look_id=look_id,
            asset_role=(
                "identity-reference"
                if kind == "actor"
                else "environment-reference"
            ),
            asset_sha256=sha256_file(asset),
            prompt_anchor=(prefix + normalized_description)[:350],
        )

    @staticmethod
    def _resolve_asset(root: Path, value: str) -> Path:
        candidate = coerce_local_path(value, base_dir=root).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"Reference asset is outside project root: {value}")
        if not candidate.exists():
            raise ValueError(f"Reference asset does not exist: {value}")
        if not candidate.is_file():
            raise ValueError(f"Reference asset is not a file: {value}")
        return candidate

    @staticmethod
    def _insert_anchor(
        destination: dict[tuple[str, str], ReferenceAnchor],
        sources: dict[tuple[str, str], str],
        anchor: ReferenceAnchor,
        source_path: str,
    ) -> None:
        key = (anchor.id, anchor.look_id)
        existing = destination.get(key)
        if existing is not None:
            qualifier = "Conflicting duplicate" if existing != anchor else "Duplicate"
            raise ValueError(
                f"{qualifier} {anchor.kind} reference "
                f"{anchor.id!r} look {anchor.look_id!r} in {source_path}; "
                f"first defined in {sources[key]}"
            )
        destination[key] = anchor
        sources[key] = source_path


def validate_project_scene_artifacts(
    project_root: str | Path,
    scenes: list[Mapping[str, Any]],
    *,
    mode: str,
    preflight_mode: PreflightMode | str,
) -> tuple[ConsistencyIssue, ...]:
    policy = PreflightMode.parse(preflight_mode)
    if policy is PreflightMode.OFF:
        return ()
    root = Path(project_root).resolve()
    issues: list[ConsistencyIssue] = []
    for scene in scenes:
        scene_number = scene.get("scene")
        if type(scene_number) is not int or scene_number <= 0:
            continue
        if mode == "ingredients":
            ingredients = scene.get("ingredients")
            ingredients = ingredients if isinstance(ingredients, Mapping) else {}
            sheet = ingredients.get("sheet_path") or scene.get(
                "ingredients_scene_sheet"
            )
            if isinstance(sheet, str) and sheet.strip():
                issue = _validate_artifact_path(
                    root,
                    sheet,
                    scene=scene_number,
                    role="ingredients_sheet",
                    policy=policy,
                )
                if issue:
                    issues.append(issue)
        elif mode == "msr":
            references = scene.get("references")
            references = references if isinstance(references, Mapping) else {}
            actor_paths = (
                references.get("actor_msr_paths")
                or references.get("actor_sheet_paths")
            )
            if isinstance(actor_paths, (list, tuple)):
                for path in actor_paths:
                    if isinstance(path, str) and path.strip():
                        issue = _validate_artifact_path(
                            root,
                            path,
                            scene=scene_number,
                            role="msr_actor",
                            policy=policy,
                        )
                        if issue:
                            issues.append(issue)
            location_path = (
                references.get("location_msr_path")
                or references.get("location_sheet_path")
            )
            if isinstance(location_path, str) and location_path.strip():
                issue = _validate_artifact_path(
                    root,
                    location_path,
                    scene=scene_number,
                    role="msr_location",
                    policy=policy,
                )
                if issue:
                    issues.append(issue)
    return tuple(issues)


def _validate_artifact_path(
    root: Path,
    value: str,
    *,
    scene: int,
    role: str,
    policy: PreflightMode,
) -> ConsistencyIssue | None:
    raw = Path(value)
    if raw.is_absolute():
        return _artifact_issue(
            f"invalid_{role}_path",
            scene,
            f"Scene {scene} {role.replace('_', ' ')} path must be project-relative",
            policy,
        )
    resolved = (root / raw).resolve()
    if not resolved.is_relative_to(root):
        return _artifact_issue(
            f"invalid_{role}_path",
            scene,
            f"Scene {scene} {role.replace('_', ' ')} path escapes the project",
            policy,
        )
    if not resolved.exists() or not resolved.is_file():
        return _artifact_issue(
            f"missing_{role}_file",
            scene,
            f"Scene {scene} {role.replace('_', ' ')} is missing or not a file",
            policy,
        )
    return None


def _artifact_issue(
    code: str,
    scene: int,
    message: str,
    policy: PreflightMode,
) -> ConsistencyIssue:
    return ConsistencyIssue(
        code=code,
        scene=scene,
        severity="error" if policy is PreflightMode.STRICT else "warning",
        message=message,
    )
