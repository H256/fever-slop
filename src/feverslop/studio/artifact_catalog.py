from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


class ArtifactCatalog:
    def __init__(self, project_root: Callable[[str], Path]):
        self.project_root = project_root

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

    @staticmethod
    def _relative_matches(root: Path, files: list[Path], predicate: Callable[[Path], bool]) -> list[str]:
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
