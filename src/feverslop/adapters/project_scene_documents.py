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


_IMAGE_EXTENSIONS = r"(?:png|jpe?g|webp)"
_VIDEO_EXTENSIONS = r"(?:mp4|mov|webm)"
_CANONICAL_STORYBOARD = re.compile(
    rf"^output/render/storyboard/scene_0*(\d+)\.{_IMAGE_EXTENSIONS}$",
    re.IGNORECASE,
)
_CANONICAL_PREVIEW = re.compile(
    rf"^output/render/scenes/scene_0*(\d+)/preview\.{_IMAGE_EXTENSIONS}$",
    re.IGNORECASE,
)
_MOVIE_STORYBOARD = re.compile(
    rf"^output/movie/storyboard/final/scene_0*(\d+)\.{_IMAGE_EXTENSIONS}$",
    re.IGNORECASE,
)
_CANONICAL_FINAL_VIDEO = re.compile(
    rf"^output/render/scenes/scene_0*(\d+)/final\.{_VIDEO_EXTENSIONS}$",
    re.IGNORECASE,
)
_LEGACY_FINAL_VIDEO = re.compile(
    rf"^output/render/(?:[^/]+/)?final/scene_0*(\d+)\.{_VIDEO_EXTENSIONS}$",
    re.IGNORECASE,
)
_LEGACY_RENDER_VIDEO = re.compile(
    rf"^output/render/ltx_(?![^/]*(?:_raw|_debug)/)[^/]+/scene_0*(\d+)\.{_VIDEO_EXTENSIONS}$",
    re.IGNORECASE,
)
_MOVIE_RENDER_VIDEO = re.compile(
    rf"^output/movie/ltx_(?:msr|ingredients|i2v|startframe_director)/scene_0*(\d+)\.{_VIDEO_EXTENSIONS}$",
    re.IGNORECASE,
)
_CANONICAL_WORKFLOW = re.compile(
    r"^output/render/scenes/scene_0*(\d+)/workflow\.json$",
    re.IGNORECASE,
)
_LEGACY_WORKFLOW = re.compile(
    r"^output/render/ltx_[^/]+_debug/scene_0*(\d+)_workflow\.json$",
    re.IGNORECASE,
)
_MOVIE_WORKFLOW = re.compile(
    r"^output/movie/ltx_(?:msr|ingredients)_debug/scene_0*(\d+)_workflow\.json$",
    re.IGNORECASE,
)
_MOVIE_INGREDIENTS_WORKFLOW = re.compile(
    r"^output/movie/ltx_ingredients/debug_workflows/scene_0*(\d+)_workflow\.json$",
    re.IGNORECASE,
)
_LEGACY_DEBUG_WORKFLOW = re.compile(
    r"^output/render/debug/scene_0*(\d+)_workflow\.json$",
    re.IGNORECASE,
)
_REJECTED_MEDIA_PREFIXES = ("reference", "ingredient")


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
        candidates: dict[int, dict[str, list[tuple[int, str]]]] = {}
        self._collect_media_candidates(
            root,
            artifacts.get("images", ()),
            candidates,
            "thumbnail_path",
            _thumbnail_candidate,
        )
        self._collect_media_candidates(
            root,
            artifacts.get("videos", ()),
            candidates,
            "video_path",
            _video_candidate,
        )
        self._collect_media_candidates(
            root,
            artifacts.get("generated_json", ()),
            candidates,
            "workflow_path",
            _workflow_candidate,
        )
        return {
            scene_number: SceneMedia(
                **{
                    field: min(field_candidates, key=lambda item: (item[0], item[1]))[1]
                    for field, field_candidates in scene_candidates.items()
                }
            )
            for scene_number, scene_candidates in candidates.items()
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
        try:
            resolved_path = path.resolve(strict=True)
        except FileNotFoundError:
            raise SceneDocumentConflict(project_id, expected_revision) from None
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
            try:
                current_path = path.resolve(strict=True)
            except FileNotFoundError:
                raise SceneDocumentConflict(project_id, expected_revision) from None
            if current_path != resolved_path:
                raise ValueError(f"Render plan path changed before write: {path}")
            try:
                actual_revision = _revision(resolved_path.read_bytes())
            except FileNotFoundError:
                raise SceneDocumentConflict(project_id, expected_revision) from None
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
    def _collect_media_candidates(
        cls,
        root: Path,
        paths: object,
        candidates: dict[int, dict[str, list[tuple[int, str]]]],
        field: str,
        candidate_for_path: Callable[[str], tuple[int, int] | None],
    ) -> None:
        for relative_path in paths:
            cls._catalogued_path(root, relative_path)
            candidate = candidate_for_path(relative_path)
            if candidate is not None:
                scene_number, rank = candidate
                candidates.setdefault(scene_number, {}).setdefault(field, []).append(
                    (rank, relative_path)
                )


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


def _thumbnail_candidate(path: str) -> tuple[int, int] | None:
    return _ranked_scene_path(
        path,
        (
            (_CANONICAL_STORYBOARD, 0),
            (_CANONICAL_PREVIEW, 1),
            (_MOVIE_STORYBOARD, 2),
        ),
    )


def _video_candidate(path: str) -> tuple[int, int] | None:
    return _ranked_scene_path(
        path,
        (
            (_CANONICAL_FINAL_VIDEO, 0),
            (_LEGACY_RENDER_VIDEO, 1),
            (_LEGACY_FINAL_VIDEO, 2),
            (_MOVIE_RENDER_VIDEO, 3),
        ),
    )


def _workflow_candidate(path: str) -> tuple[int, int] | None:
    return _ranked_scene_path(
        path,
        (
            (_CANONICAL_WORKFLOW, 0),
            (_LEGACY_WORKFLOW, 1),
            (_MOVIE_WORKFLOW, 2),
            (_MOVIE_INGREDIENTS_WORKFLOW, 3),
            (_LEGACY_DEBUG_WORKFLOW, 4),
        ),
    )


def _ranked_scene_path(
    path: str,
    patterns: tuple[tuple[re.Pattern[str], int], ...],
) -> tuple[int, int] | None:
    normalized = Path(path).as_posix()
    parts = (part.lower() for part in Path(normalized).parts)
    if any(part.startswith(_REJECTED_MEDIA_PREFIXES) for part in parts):
        return None
    for pattern, rank in patterns:
        match = pattern.search(normalized)
        if match:
            return int(match.group(1)), rank
    return None
