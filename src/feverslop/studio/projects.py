from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from feverslop.domain.slug_utils import (
    slugify_project_name,  # noqa: F401 -- re-exported for backward compatibility
)
from feverslop.ports.reporting import Reporter
from feverslop.studio.artifact_locking import artifact_write_lock

_logger = logging.getLogger(__name__)


class StudioPathError(ValueError):
    pass


class ArtifactConflict(ValueError):
    def __init__(self, path: str, expected_revision: str | None, actual_revision: str | None):
        if expected_revision is None and actual_revision is not None:
            message = f"Artifact already exists; load target before saving: {path}"
        else:
            message = f"Artifact changed: {path}"
        super().__init__(message)
        self.path = path
        self.expected_revision = expected_revision
        self.actual_revision = actual_revision


@dataclass(frozen=True)
class ArtifactRequest:
    path: str
    data: Any
    expected_revision: str | None = None
    create_only: bool = False


@dataclass(frozen=True)
class RenderPlanPatch:
    path: str
    scene: int
    updates: dict[str, Any]
    expected_revision: str | None = None


@dataclass(frozen=True)
class ProjectCreateRequest:
    project_type: str
    name: str
    silent_mode: bool = False
    source_type: str = "short_story"
    story_text: str = ""
    desired_length: float = 60.0
    dialogue_language: str = "English"
    movie_mode: str = "scaffold"
    idea: str = ""
    song_style: str = ""
    duration_seconds: float = 120.0
    width: int = 1280
    height: int = 704
    fps: int = 24
    pipeline_mode: str = "classic"
    movie_planner_backend: str = "llm"
    movie_reference_backend: str = "comfyui"
    movie_render_backend: str = "comfyui"
    movie_hero_workflow: str = "workflows/image_t2i_startframe_krea_v1.json"
    movie_edit_workflow: str = "workflows/image_edit_flux2_klein_1ref_v1.json"
    movie_startframe_director_backend: str = "krea2"
    movie_director_workflow: str = "workflows/image_t2i_startframe_krea_v1.json"
    movie_mask_workflow: str = "workflows/image_mask_sam3_actor_regions_v1.json"
    movie_identity_repair_workflow: str = "workflows/image_repair_sdxl_ipadapter_identity_v1.json"
    movie_detail_workflow: str = "workflows/image_detail_easyuse_startframe_v1.json"
    movie_startframe_comfyui_base_url: str = "http://localhost:8188"
    movie_startframe_write_debug_workflows: bool = False
    movie_startframe_debug_workflows_dir: str = ""
    movie_startframe_validator_base_url: str = "http://your-llm-server.local/v1"
    movie_startframe_validator_model: str = "gemma4-26b-a4b:vision"
    movie_msr_workflow: str = "workflows/video_default_ltxv_msr_1actor_1background_v4.json"
    movie_msr_i2v_workflow: str = "workflows/video_default_i2v_ltxv_msr_1actor_1background_v4.json"
    movie_i2v_workflow: str = "workflows/video_ltxv_i2v_v2.json"
    movie_video_workflow: str = "msr"
    movie_continuity_keyframes: str = "none"
    movie_refine_location_prompts: bool = False
    movie_refine_actor_prompts: bool = False


AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
AUDIO_MIME_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/flac",
    "audio/x-flac",
    "audio/mp4",
    "audio/x-m4a",
    "audio/ogg",
}


def sanitize_audio_filename(value: str) -> str:
    name = re.split(r"[\\/]+", str(value or "").strip())[-1]
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not name:
        raise ValueError("Audio filename is required")
    return name


class ProjectStore:
    def __init__(
        self,
        projects_root: str | Path = "projects",
        reporter: Reporter | None = None,
        max_upload_size: int = 100 * 1024 * 1024,
    ):
        self.projects_root = Path(projects_root).resolve()
        from feverslop.studio.artifact_catalog import ArtifactCatalog
        from feverslop.studio.media_store import MediaStore
        from feverslop.studio.pipeline_state_store import PipelineStateStore
        from feverslop.studio.project_repository import ProjectRepository

        self.repository = ProjectRepository(
            projects_root=self.projects_root,
            project_root=self.project_root,
            read_json_file=lambda path: self._read_json_file(path, default={}),
            reporter=reporter,
        )
        self.artifact_catalog = ArtifactCatalog(self.project_root)
        self.media_store = MediaStore(
            self.project_root,
            self.resolve_project_path,
            lambda path: self._read_json_file(path, default={}),
            max_upload_size=max_upload_size,
        )
        self.pipeline_state_store = PipelineStateStore(
            self.project_root,
            lambda path: self._read_json_file(path, default={}),
        )

    def list_projects(self) -> list[dict[str, Any]]:
        if not self.projects_root.exists():
            return []
        projects = [self.describe_project(path.name) for path in self.projects_root.iterdir() if path.is_dir()]
        return sorted(projects, key=lambda project: project["id"])

    def describe_project(self, project_id: str) -> dict[str, Any]:
        root = self.project_root(project_id)
        config_path = root / "config.json"
        config = self._read_json_file(config_path, default={})
        metadata = self.project_metadata(project_id)
        name = str(metadata.get("display_name") or config.get("project_name") or project_id)
        catalog = self.artifact_catalog.catalog_snapshot(project_id)
        artifacts = catalog["artifacts"]
        is_movie = metadata.get("project_type") == "movie"
        return {
            "id": project_id,
            "name": name,
            "path": root.as_posix(),
            "project_type": metadata.get("project_type", "standard_music_video"),
            "silent_mode": self._silent_mode(config, metadata),
            "metadata": metadata,
            "status": {
                "config": "present" if config_path.exists() or (is_movie and metadata) else "missing",
                "render_plan": "present" if artifacts["render_plans"] else "missing",
                "references": "present" if artifacts["references"] else "missing",
                "videos": "present" if artifacts["videos"] else "missing",
            },
            "artifacts": artifacts,
            "artifact_sizes": catalog["artifact_sizes"],
        }

    def create_project(self, request: ProjectCreateRequest) -> dict[str, Any]:
        slug = self.repository.create_project(request)
        return self.describe_project(slug)

    def project_metadata(self, project_id: str) -> dict[str, Any]:
        return self.repository.project_metadata(project_id)

    def list_artifacts(self, project_id: str) -> dict[str, list[str]]:
        return self.artifact_catalog.list_artifacts(project_id)

    def artifact_sizes(self, project_id: str) -> dict[str, Any]:
        return self.artifact_catalog.artifact_sizes(project_id)

    def read_artifact(self, project_id: str, path: str) -> dict[str, Any]:
        artifact_path = self.resolve_project_path(project_id, path)
        if not artifact_path.exists():
            return {"path": path, "data": None, "exists": False, "revision": None}
        payload = artifact_path.read_bytes()
        return {
            "path": path,
            "data": json.loads(payload.decode("utf-8-sig")),
            "exists": True,
            "revision": _artifact_revision(payload),
        }

    def write_artifact(self, project_id: str, request: ArtifactRequest) -> dict[str, Any]:
        artifact_path = self.resolve_project_path(project_id, request.path)
        data = request.data
        if artifact_path.name == "config.json":
            from feverslop.studio.project_validation import validate_project_config

            metadata = self.project_metadata(project_id)
            validate_project_config(data, project_type=str(metadata.get("project_type") or "standard_music_video"))
            if isinstance(data, dict) and data.get("silent_mode") is None:
                data = {**data, "silent_mode": False}
        payload = (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        with artifact_write_lock(artifact_path):
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            if request.create_only:
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{artifact_path.name}.",
                    suffix=".tmp",
                    dir=artifact_path.parent,
                )
                temporary_path = Path(temporary_name)
                try:
                    with os.fdopen(descriptor, "wb") as temporary:
                        temporary.write(payload)
                        temporary.flush()
                        os.fsync(temporary.fileno())
                    try:
                        os.link(temporary_path, artifact_path)
                    except FileExistsError:
                        actual = _path_revision(artifact_path)
                        raise ArtifactConflict(request.path, None, actual) from None
                    except OSError as exc:
                        raise OSError(
                            f"Could not atomically create artifact: {request.path}",
                        ) from exc
                finally:
                    temporary_path.unlink(missing_ok=True)
            else:
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{artifact_path.name}.",
                    suffix=".tmp",
                    dir=artifact_path.parent,
                )
                temporary_path = Path(temporary_name)
                try:
                    with os.fdopen(descriptor, "wb") as temporary:
                        temporary.write(payload)
                        temporary.flush()
                        os.fsync(temporary.fileno())
                    if request.expected_revision is not None:
                        actual = _path_revision(artifact_path)
                        if actual != request.expected_revision:
                            raise ArtifactConflict(
                                request.path,
                                request.expected_revision,
                                actual,
                            )
                    temporary_path.replace(artifact_path)
                finally:
                    temporary_path.unlink(missing_ok=True)
        return {
            "path": request.path,
            "data": data,
            "exists": True,
            "revision": _artifact_revision(payload),
        }

    def write_media_data_url(self, project_id: str, path: str, data_url: str) -> dict[str, str]:
        return self.media_store.write_media_data_url(project_id, path, data_url)

    def store_audio_upload(self, project_id: str, filename: str, content_type: str, source) -> dict[str, str]:
        return self.media_store.store_audio_upload(project_id, filename, content_type, source)

    def patch_render_plan(self, project_id: str, patch: RenderPlanPatch) -> dict[str, Any]:
        artifact_path = self.resolve_project_path(project_id, patch.path)
        with artifact_write_lock(artifact_path):
            render_plan = self._read_json_file(artifact_path, default=[])
            if not isinstance(render_plan, list):
                raise ValueError("Render plan must be a JSON array")
            for scene in render_plan:
                if int(scene.get("scene", -1)) == patch.scene:
                    scene.update(patch.updates)
                    written = self.write_artifact(
                        project_id,
                        ArtifactRequest(
                            path=patch.path,
                            data=render_plan,
                            expected_revision=patch.expected_revision,
                        ),
                    )
                    return {
                        "path": patch.path,
                        "scene": scene,
                        "revision": written["revision"],
                    }
        raise KeyError(f"Scene {patch.scene} not found in {patch.path}")

    def project_root(self, project_id: str) -> Path:
        root = (self.projects_root / project_id).resolve()
        if root.parent != self.projects_root:
            raise StudioPathError("Project id must name a direct child of projects root")
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Project not found: {project_id}")
        return root

    @staticmethod
    def _write_project_metadata(root: Path, metadata: dict[str, Any]) -> None:
        path = root / ".studio" / "project.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def resolve_project_path(self, project_id: str, path: str) -> Path:
        root = self.project_root(project_id)
        target = (root / path).resolve()
        if not target.is_relative_to(root):
            raise StudioPathError("Path escapes project root")
        return target

    def resolve_media_path(self, project_id: str, path: str) -> Path:
        media_path = self.resolve_project_path(project_id, path)
        if media_path.suffix.lower() not in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".gif",
            ".mp4",
            ".mov",
            ".webm",
            ".mp3",
            ".wav",
            ".m4a",
            ".flac",
            ".ogg",
        }:
            raise StudioPathError("Unsupported media type")
        if not media_path.exists():
            raise FileNotFoundError(path)
        return media_path

    def resolve_video_path(self, project_id: str, path: str) -> Path:
        video_path = self.resolve_project_path(project_id, path)
        if video_path.suffix.lower() not in {".mp4", ".mov", ".webm"}:
            raise StudioPathError("Unsupported video type")
        if not video_path.exists():
            raise FileNotFoundError(path)
        return video_path

    def thumbnail_cache_path(self, project_id: str, key: str) -> Path:
        root = self.project_root(project_id)
        cache = root / ".studio" / "thumbnails"
        cache.mkdir(parents=True, exist_ok=True)
        return cache / f"{key}.jpg"

    def clear_thumbnail_cache(self, project_id: str) -> int:
        cache = self.project_root(project_id) / ".studio" / "thumbnails"
        if not cache.exists():
            return 0
        count = sum(1 for path in cache.rglob("*") if path.is_file())
        shutil.rmtree(cache)
        return count

    def record_pipeline_run(self, project_id: str, *, action: str, stages: list[str], status: str) -> dict[str, Any]:
        return self.pipeline_state_store.record_pipeline_run(project_id, action=action, stages=stages, status=status)

    @staticmethod
    def _silent_mode(config: Any, metadata: Any) -> bool:
        if isinstance(config, dict) and isinstance(config.get("silent_mode", False), bool):
            return bool(config.get("silent_mode", False))
        if isinstance(metadata, dict) and isinstance(metadata.get("silent_mode", False), bool):
            return bool(metadata.get("silent_mode", False))
        return False

    @staticmethod
    def _read_json_file(path: Path, *, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            _logger.warning("Could not parse JSON artifact %s; using default: invalid JSON (%s)", path, exc)
            return default


def _artifact_revision(payload: bytes) -> str:
    """Compute SHA-256 of artifact payload, normalizing UTF-8 BOM."""
    normalized = payload
    if normalized.startswith(b"\xef\xbb\xbf"):
        normalized = normalized[3:]
    return hashlib.sha256(normalized).hexdigest()


def _path_revision(path: Path) -> str | None:
    try:
        return _artifact_revision(path.read_bytes())
    except FileNotFoundError:
        return None
