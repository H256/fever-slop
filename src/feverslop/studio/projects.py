from __future__ import annotations

import json
import base64
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
        name = str(config.get("project_name") or project_id)
        artifacts = self.list_artifacts(project_id)
        return {
            "id": project_id,
            "name": name,
            "path": root.as_posix(),
            "status": {
                "config": "present" if config_path.exists() else "missing",
                "render_plan": "present" if artifacts["render_plans"] else "missing",
                "references": "present" if artifacts["references"] else "missing",
                "videos": "present" if artifacts["videos"] else "missing",
            },
            "artifacts": artifacts,
        }

    def list_artifacts(self, project_id: str) -> dict[str, list[str]]:
        root = self.project_root(project_id)
        files = [path for path in root.rglob("*") if path.is_file()]
        return {
            "configs": self._relative_matches(root, files, lambda path: path.name == "config.json"),
            "render_plans": self._relative_matches(root, files, lambda path: path.name.startswith("render_plan") and path.suffix == ".json"),
            "references": self._relative_matches(root, files, lambda path: "reference" in path.as_posix() and path.suffix.lower() in {".json", ".png", ".jpg", ".jpeg", ".webp"}),
            "generated_json": self._relative_matches(root, files, lambda path: path.suffix == ".json" and path.name != "config.json"),
            "videos": self._relative_matches(root, files, lambda path: path.suffix.lower() in {".mp4", ".mov", ".webm"}),
            "images": self._relative_matches(root, files, lambda path: path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}),
        }

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

    def resolve_project_path(self, project_id: str, path: str) -> Path:
        root = self.project_root(project_id)
        target = (root / path).resolve()
        if not target.is_relative_to(root):
            raise StudioPathError("Path escapes project root")
        return target

    def resolve_media_path(self, project_id: str, path: str) -> Path:
        media_path = self.resolve_project_path(project_id, path)
        if media_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".mov", ".webm", ".mp3", ".wav"}:
            raise StudioPathError("Unsupported media type")
        if not media_path.exists():
            raise FileNotFoundError(path)
        return media_path

    @staticmethod
    def _read_json_file(path: Path, *, default: Any) -> Any:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8-sig"))

    @staticmethod
    def _relative_matches(root: Path, files: list[Path], predicate) -> list[str]:
        return sorted(path.relative_to(root).as_posix() for path in files if predicate(path))
