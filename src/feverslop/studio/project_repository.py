from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from feverslop.adapters.movie_planning import DeterministicMoviePlanner
from feverslop.application.movie import MovieInput, ScaffoldMovieUseCase
from feverslop.studio.project_validation import validate_full_auto_inputs
from feverslop.studio.projects import ProjectCreateRequest, StudioPathError, slugify_project_name


class ProjectRepository:
    def __init__(
        self,
        *,
        projects_root: Path,
        project_root: Callable[[str], Path],
        read_json_file: Callable[[Path], Any],
    ):
        self.projects_root = projects_root
        self.project_root = project_root
        self.read_json_file = read_json_file

    def create_project(self, request: ProjectCreateRequest) -> str:
        project_type = str(request.project_type or "").strip()
        if project_type not in {"standard_music_video", "full_auto", "movie"}:
            raise ValueError("project_type must be standard_music_video, full_auto, or movie")
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
            validate_full_auto_inputs(request)
        if project_type == "movie":
            return self._create_movie_project(request, slug)

        root.mkdir(parents=True)
        metadata = {
            "project_type": project_type,
            "display_name": name,
            "slug": slug,
            "silent_mode": bool(request.silent_mode),
        }
        if project_type == "full_auto":
            metadata["full_auto"] = {
                "idea": str(request.idea).strip(),
                "song_style": str(request.song_style).strip(),
                "duration_seconds": float(request.duration_seconds),
                "width": int(request.width),
                "height": int(request.height),
                "fps": int(request.fps),
                "pipeline_mode": str(request.pipeline_mode or "classic"),
            }
        self.write_project_metadata(root, metadata)
        if project_type == "standard_music_video":
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "project_name": name,
                        "input_audio": "",
                        "silent_mode": bool(request.silent_mode),
                        "audio": {"language": "en"},
                        "scene_generation": {"seed": -1},
                    },
                    indent=2,
                    ensure_ascii=False,
                ) + "\n",
                encoding="utf-8",
            )
        return slug

    def _create_movie_project(self, request: ProjectCreateRequest, slug: str) -> str:
        result = ScaffoldMovieUseCase(
            planner=DeterministicMoviePlanner(),
            projects_root=self.projects_root,
        ).execute(
            MovieInput(
                name=str(request.name).strip(),
                source_type=str(request.source_type or "short_story"),
                story_text=str(request.story_text or ""),
                desired_length=float(request.desired_length),
                width=int(request.width),
                height=int(request.height),
                mode=str(request.movie_mode or "scaffold"),
            )
        )
        if result.project_slug != slug:
            raise ValueError("Movie project slug mismatch")
        return slug

    def project_metadata(self, project_id: str) -> dict[str, Any]:
        root = self.project_root(project_id)
        metadata = self.read_json_file(root / ".studio" / "project.json")
        if metadata:
            return metadata
        config = self.read_json_file(root / "config.json")
        return {
            "project_type": "standard_music_video",
            "display_name": str(config.get("project_name") or project_id) if isinstance(config, dict) else project_id,
            "slug": project_id,
            "silent_mode": bool(config.get("silent_mode", False)) if isinstance(config, dict) and isinstance(config.get("silent_mode", False), bool) else False,
        }

    @staticmethod
    def write_project_metadata(root: Path, metadata: dict[str, Any]) -> None:
        path = root / ".studio" / "project.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
