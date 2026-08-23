from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from feverslop.scene_artifacts import SceneArtifactLayout


class ArtifactCatalog:
    def __init__(self, project_root: Callable[[str], Path]):
        self.project_root = project_root

    def list_artifacts(self, project_id: str) -> dict[str, list[str]]:
        return self.catalog_snapshot(project_id)["artifacts"]

    def artifact_sizes(self, project_id: str) -> dict[str, Any]:
        return self.catalog_snapshot(project_id)["artifact_sizes"]

    def catalog_snapshot(self, project_id: str) -> dict[str, Any]:
        root = self.project_root(project_id)
        layout = SceneArtifactLayout(root)
        files = [path for path in root.rglob("*") if path.is_file() and ".studio" not in path.relative_to(root).parts]
        render_plans = [path for path in files if self._is_render_plan(path, layout)]
        render_plans.sort(key=lambda path: (path.parent != layout.plans_dir, path.as_posix()))
        artifacts = {
            "configs": self._relative_matches(root, files, lambda path: path.name == "config.json"),
            "render_plans": [path.relative_to(root).as_posix() for path in render_plans],
            "references": self._relative_matches(root, files, lambda path: "reference" in path.as_posix() and path.suffix.lower() in {".json", ".png", ".jpg", ".jpeg", ".webp"}),
            "generated_json": self._relative_matches(root, files, lambda path: path.suffix == ".json" and path.name != "config.json"),
            "videos": self._relative_matches(root, files, lambda path: path.suffix.lower() in {".mp4", ".mov", ".webm"}),
            "images": self._relative_matches(root, files, lambda path: path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}),
            "audio": self._relative_matches(root, files, lambda path: path.suffix.lower() in {".mp3", ".wav", ".m4a", ".flac", ".ogg"}),
        }
        totals = dict.fromkeys(["configs", "render_plans", "references", "generated_json", "videos", "images", "audio", "other"], 0)
        for path in files:
            try:
                totals[self._artifact_size_group(path, layout)] += path.stat().st_size
            except OSError:
                pass  # broken symlink or permission denied — default to 0 bytes
        return {"artifacts": artifacts, "artifact_sizes": {"total_bytes": sum(totals.values()), "by_type": totals}}

    @staticmethod
    def _relative_matches(root: Path, files: list[Path], predicate: Callable[[Path], bool]) -> list[str]:
        return sorted(path.relative_to(root).as_posix() for path in files if predicate(path))

    @staticmethod
    def _is_render_plan(path: Path, layout: SceneArtifactLayout) -> bool:
        return path.suffix == ".json" and (path.parent == layout.plans_dir or path.name.startswith("render_plan"))

    @staticmethod
    def _artifact_size_group(path: Path, layout: SceneArtifactLayout) -> str:
        suffix = path.suffix.lower()
        posix = path.as_posix()
        if path.name == "config.json":
            return "configs"
        if ArtifactCatalog._is_render_plan(path, layout):
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
