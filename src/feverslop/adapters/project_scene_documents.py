from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from feverslop.domain.scene_workspace import SceneMedia
from feverslop.ports.scene_documents import SceneDocumentConflict, SceneDocumentSnapshot
from feverslop.studio.artifact_catalog import ArtifactCatalog


class _ArtifactCatalog(Protocol):
    def list_artifacts(self, project_id: str) -> dict[str, list[str]]: ...


_SCENE_PATH_PATTERN = re.compile(r"(?:^|/)scene[_-]?0*(\d+)(?:/|[_\-.])", re.IGNORECASE)


class ProjectSceneDocuments:
    """Adapt catalogued project artifacts to the scene workspace ports."""

    def __init__(
        self,
        project_root: Callable[[str], Path],
        *,
        catalog: _ArtifactCatalog | None = None,
    ) -> None:
        self._project_root = project_root
        self._catalog = catalog or ArtifactCatalog(project_root)

    def load(self, project_id: str) -> SceneDocumentSnapshot:
        root, path = self._render_plan_path(project_id)
        payload = self._read_catalogued_file(root, path)
        scenes = self._decode_render_plan(payload, path)
        return SceneDocumentSnapshot(
            scenes=tuple(scenes),
            revision=_revision(payload),
        )

    def patch_scene(
        self,
        project_id: str,
        scene_number: int,
        changes: Mapping[str, object],
        expected_revision: str,
    ) -> SceneDocumentSnapshot:
        root, path = self._render_plan_path(project_id)
        try:
            payload = self._read_catalogued_file(root, path)
        except FileNotFoundError:
            raise SceneDocumentConflict(project_id, expected_revision) from None
        actual_revision = _revision(payload)
        if actual_revision != expected_revision:
            raise SceneDocumentConflict(project_id, expected_revision, actual_revision)

        scenes = self._decode_render_plan(payload, path)
        scene = next(
            (
                item
                for item in scenes
                if isinstance(item, dict) and _scene_number(item) == scene_number
            ),
            None,
        )
        if scene is None:
            raise KeyError(f"Scene {scene_number} not found in {path}")
        _merge_json_object(scene, changes)

        encoded = (json.dumps(scenes, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        self._atomic_replace(
            root,
            path,
            encoded,
            project_id=project_id,
            expected_revision=expected_revision,
        )
        return SceneDocumentSnapshot(scenes=tuple(scenes), revision=_revision(encoded))

    def load_media(self, project_id: str) -> Mapping[int, SceneMedia]:
        root = self._resolved_root(project_id)
        artifacts = self._catalog.list_artifacts(project_id)
        media: dict[int, dict[str, str]] = {}
        self._collect_media(root, artifacts.get("images", ()), media, "thumbnail_path")
        self._collect_media(root, artifacts.get("videos", ()), media, "video_path")
        workflows = (
            path
            for path in artifacts.get("generated_json", ())
            if _is_workflow_path(path)
        )
        self._collect_media(root, workflows, media, "workflow_path")
        return {
            scene_number: SceneMedia(**paths)
            for scene_number, paths in media.items()
        }

    def _render_plan_path(self, project_id: str) -> tuple[Path, Path]:
        root = self._resolved_root(project_id)
        artifacts = self._catalog.list_artifacts(project_id)
        render_plans = artifacts.get("render_plans", ())
        if not render_plans:
            raise FileNotFoundError(f"No render plan found for project {project_id!r}")
        return root, self._catalogued_path(root, render_plans[0])

    def _resolved_root(self, project_id: str) -> Path:
        root = Path(self._project_root(project_id)).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Project not found: {project_id}")
        return root

    @staticmethod
    def _catalogued_path(root: Path, relative_path: str) -> Path:
        candidate = (root / relative_path).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"Catalog artifact is outside project root: {relative_path}")
        return candidate

    @staticmethod
    def _read_catalogued_file(root: Path, path: Path) -> bytes:
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError(f"Catalog artifact is outside project root: {path}")
        return resolved.read_bytes()

    @staticmethod
    def _decode_render_plan(payload: bytes, path: Path) -> list[dict[str, Any]]:
        try:
            decoded = json.loads(payload.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed render plan JSON: {path}") from exc
        if not isinstance(decoded, list):
            raise ValueError(f"Render plan must be a JSON array: {path}")
        for index, scene in enumerate(decoded):
            if not isinstance(scene, dict):
                raise ValueError(f"Render plan scene at index {index} must be a JSON object: {path}")
        return decoded

    @classmethod
    def _atomic_replace(
        cls,
        root: Path,
        path: Path,
        payload: bytes,
        *,
        project_id: str,
        expected_revision: str,
    ) -> None:
        resolved_path = path.resolve(strict=True)
        if not resolved_path.is_relative_to(root):
            raise ValueError(f"Render plan is outside project root: {path}")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if path.resolve(strict=True) != resolved_path:
                raise ValueError(f"Render plan path changed before write: {path}")
            actual_revision = _revision(resolved_path.read_bytes())
            if actual_revision != expected_revision:
                raise SceneDocumentConflict(
                    project_id,
                    expected_revision,
                    actual_revision,
                )
            temporary_path.replace(resolved_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @classmethod
    def _collect_media(
        cls,
        root: Path,
        paths: object,
        media: dict[int, dict[str, str]],
        field: str,
    ) -> None:
        for relative_path in paths:
            cls._catalogued_path(root, relative_path)
            scene_number = _scene_number_from_path(relative_path)
            if scene_number is not None:
                media.setdefault(scene_number, {}).setdefault(field, relative_path)


def _revision(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _scene_number(scene: Mapping[str, Any]) -> int | None:
    try:
        return int(scene.get("scene"))
    except (TypeError, ValueError):
        return None


def _merge_json_object(target: dict[str, Any], changes: Mapping[str, object]) -> None:
    for key, value in changes.items():
        current = target.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            _merge_json_object(current, value)
        else:
            target[key] = _json_copy(value)


def _json_copy(value: object) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_copy(item) for item in value]
    return value


def _scene_number_from_path(path: str) -> int | None:
    match = _SCENE_PATH_PATTERN.search(Path(path).as_posix())
    return int(match.group(1)) if match else None


def _is_workflow_path(path: str) -> bool:
    stem = Path(path).stem.lower()
    return stem == "workflow" or stem.endswith("_workflow")
