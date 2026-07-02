from __future__ import annotations

import base64
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class StudioPathError(ValueError):
    pass


@dataclass(frozen=True)
class ArtifactRequest:
    path: str
    data: Any


@dataclass(frozen=True)
class RenderPlanPatch:
    path: str
    scene: int
    updates: dict[str, Any]


@dataclass(frozen=True)
class ProjectCreateRequest:
    project_type: str
    name: str
    idea: str = ""
    song_style: str = ""


def slugify_project_name(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return re.sub(r"-+", "-", slug)


class ProjectStore:
    def __init__(self, projects_root: str | Path = "projects"):
        self.projects_root = Path(projects_root).resolve()

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
        artifacts = self.list_artifacts(project_id)
        return {
            "id": project_id,
            "name": name,
            "path": root.as_posix(),
            "project_type": metadata.get("project_type", "standard_music_video"),
            "metadata": metadata,
            "status": {
                "config": "present" if config_path.exists() else "missing",
                "render_plan": "present" if artifacts["render_plans"] else "missing",
                "references": "present" if artifacts["references"] else "missing",
                "videos": "present" if artifacts["videos"] else "missing",
            },
            "artifacts": artifacts,
            "artifact_sizes": self.artifact_sizes(project_id),
        }

    def create_project(self, request: ProjectCreateRequest) -> dict[str, Any]:
        project_type = str(request.project_type or "").strip()
        if project_type not in {"standard_music_video", "full_auto"}:
            raise ValueError("project_type must be standard_music_video or full_auto")
        name = str(request.name or "").strip()
        if not name:
            raise ValueError("Project name is required")
        slug = slugify_project_name(name)
        if not slug:
            raise ValueError("Project slug is empty after slugifying the name")
        root = (self.projects_root / slug).resolve()
        if root.parent != self.projects_root:
            raise StudioPathError("Project id must name a direct child of projects root")
        if root.exists():
            raise ValueError(f"Project already exists: {slug}")
        if project_type == "full_auto":
            if not str(request.idea or "").strip():
                raise ValueError("Full-auto idea is required")
            if not str(request.song_style or "").strip():
                raise ValueError("Full-auto song style is required")

        root.mkdir(parents=True)
        metadata = {
            "project_type": project_type,
            "display_name": name,
            "slug": slug,
        }
        if project_type == "full_auto":
            metadata["full_auto"] = {
                "idea": str(request.idea).strip(),
                "song_style": str(request.song_style).strip(),
            }
        self._write_project_metadata(root, metadata)
        if project_type == "standard_music_video":
            (root / "config.json").write_text(
                json.dumps({"project_name": name, "input_audio": ""}, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        return self.describe_project(slug)

    def project_metadata(self, project_id: str) -> dict[str, Any]:
        root = self.project_root(project_id)
        metadata = self._read_json_file(root / ".studio" / "project.json", default={})
        if metadata:
            return metadata
        config = self._read_json_file(root / "config.json", default={})
        return {
            "project_type": "standard_music_video",
            "display_name": str(config.get("project_name") or project_id),
            "slug": project_id,
        }

    def list_artifacts(self, project_id: str) -> dict[str, list[str]]:
        root = self.project_root(project_id)
        files = [path for path in root.rglob("*") if path.is_file() and ".studio" not in path.relative_to(root).parts]
        return {
            "configs": self._relative_matches(root, files, lambda path: path.name == "config.json"),
            "render_plans": self._relative_matches(root, files, lambda path: path.name.startswith("render_plan") and path.suffix == ".json"),
            "references": self._relative_matches(root, files, lambda path: "reference" in path.as_posix() and path.suffix.lower() in {".json", ".png", ".jpg", ".jpeg", ".webp"}),
            "generated_json": self._relative_matches(root, files, lambda path: path.suffix == ".json" and path.name != "config.json"),
            "videos": self._relative_matches(root, files, lambda path: path.suffix.lower() in {".mp4", ".mov", ".webm"}),
            "images": self._relative_matches(root, files, lambda path: path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}),
            "audio": self._relative_matches(root, files, lambda path: path.suffix.lower() in {".mp3", ".wav", ".m4a", ".flac", ".ogg"}),
        }

    def artifact_sizes(self, project_id: str) -> dict[str, Any]:
        root = self.project_root(project_id)
        totals = {key: 0 for key in ["configs", "render_plans", "references", "generated_json", "videos", "images", "audio", "other"]}
        for path in (path for path in root.rglob("*") if path.is_file() and ".studio" not in path.relative_to(root).parts):
            totals[self._artifact_size_group(path)] += path.stat().st_size
        return {"total_bytes": sum(totals.values()), "by_type": totals}

    def read_artifact(self, project_id: str, path: str) -> dict[str, Any]:
        artifact_path = self.resolve_project_path(project_id, path)
        return {"path": path, "data": self._read_json_file(artifact_path, default=None)}

    def write_artifact(self, project_id: str, request: ArtifactRequest) -> dict[str, Any]:
        artifact_path = self.resolve_project_path(project_id, request.path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(request.data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"path": request.path, "data": request.data}

    def write_media_data_url(self, project_id: str, path: str, data_url: str) -> dict[str, str]:
        media_path = self.resolve_project_path(project_id, path)
        if media_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise StudioPathError("Unsupported uploaded media type")
        header, separator, encoded = data_url.partition(",")
        if not separator or not header.startswith("data:image/"):
            raise StudioPathError("Expected an image data URL")
        media_path.parent.mkdir(parents=True, exist_ok=True)
        media_path.write_bytes(base64.b64decode(encoded))
        return {"path": path}

    def patch_render_plan(self, project_id: str, patch: RenderPlanPatch) -> dict[str, Any]:
        artifact_path = self.resolve_project_path(project_id, patch.path)
        render_plan = self._read_json_file(artifact_path, default=[])
        if not isinstance(render_plan, list):
            raise ValueError("Render plan must be a JSON array")
        for scene in render_plan:
            if int(scene.get("scene", -1)) == patch.scene:
                scene.update(patch.updates)
                artifact_path.write_text(json.dumps(render_plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                return {"path": patch.path, "scene": scene}
        raise KeyError(f"Scene {patch.scene} not found in {patch.path}")

    def project_root(self, project_id: str) -> Path:
        root = (self.projects_root / project_id).resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Project not found: {project_id}")
        if root.parent != self.projects_root:
            raise StudioPathError("Project id must name a direct child of projects root")
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

    @staticmethod
    def _read_json_file(path: Path, *, default: Any) -> Any:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8-sig"))

    @staticmethod
    def _relative_matches(root: Path, files: list[Path], predicate) -> list[str]:
        return sorted(path.relative_to(root).as_posix() for path in files if predicate(path))

    @staticmethod
    def _artifact_size_group(path: Path) -> str:
        suffix = path.suffix.lower()
        posix = path.as_posix()
        if path.name == "config.json":
            return "configs"
        if path.name.startswith("render_plan") and suffix == ".json":
            return "render_plans"
        if "reference" in posix and suffix in {".json", ".png", ".jpg", ".jpeg", ".webp"}:
            return "references"
        if suffix in {".mp4", ".mov", ".webm"}:
            return "videos"
        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            return "images"
        if suffix in {".mp3", ".wav", ".m4a", ".flac", ".ogg"}:
            return "audio"
        if suffix == ".json":
            return "generated_json"
        return "other"
