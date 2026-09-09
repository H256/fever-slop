from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from feverslop.adapters.movie_artifact_writer import LocalMovieArtifactWriter
from feverslop.application.movie import (
    MovieInput,
    ScaffoldMovieUseCase,
    _dialogue_language,
    _render_profile_config,
    movie_project_config,
)
from feverslop.composition.movie_planner import build_movie_planner
from feverslop.config.project_config import (
    SCENE_PROMPT_WORD_COUNT_MAX,
    SCENE_PROMPT_WORD_COUNT_MIN,
    VIDEO_PIPELINE_BY_MODE,
    validate_full_auto_inputs,
    validate_pipeline_mode,
)
from feverslop.domain.slug_utils import slugify_project_name
from feverslop.ports.reporting import Reporter
from feverslop.ports.project_requests import ProjectCreateRequest, StudioPathError
from feverslop.utils.io import atomic_write_json


class ProjectRepository:
    def __init__(
        self,
        *,
        projects_root: Path,
        project_root: Callable[[str], Path],
        read_json_file: Callable[[Path], Any],
        reporter: Reporter | None = None,
    ):
        self.projects_root = projects_root
        self.project_root = project_root
        self.read_json_file = read_json_file
        self.reporter = reporter

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
        pipeline_mode = "classic"
        if project_type != "movie":
            pipeline_mode = validate_pipeline_mode(request.pipeline_mode)
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
                "silent_mode": bool(request.silent_mode),
                "pipeline_mode": pipeline_mode,
            }
        self.write_project_metadata(root, metadata)
        if project_type == "standard_music_video":
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "project_name": name,
                        "input_audio": "",
                        "silent_mode": bool(request.silent_mode),
                        "video_pipeline": VIDEO_PIPELINE_BY_MODE[pipeline_mode],
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
        movie_config = movie_project_config(request)
        config = movie_default_config(request)
        result = ScaffoldMovieUseCase(
            planner=build_movie_planner(movie_config),
            projects_root=self.projects_root,
            artifact_writer=LocalMovieArtifactWriter(),
            reporter=self.reporter,
        ).execute(
            MovieInput(
                name=str(request.name).strip(),
                source_type=str(request.source_type or "short_story"),
                story_text=str(request.story_text or ""),
                desired_length=float(request.desired_length),
                width=int(request.width),
                height=int(request.height),
                mode=str(request.movie_mode or "scaffold"),
                min_scene_duration=float(config["scene_generation"]["min_duration"]),
                max_scene_duration=float(config["scene_generation"]["max_duration"]),
                config={**config, **movie_config},
            ),
        )
        if result.project_slug != slug:
            raise ValueError("Movie project slug mismatch")
        metadata = self.project_metadata(slug)
        metadata["movie"] = {**dict(metadata.get("movie") or {}), **movie_config}
        self.write_project_metadata(result.project_dir, metadata)
        return slug

    def project_metadata(self, project_id: str) -> dict[str, Any]:
        root = self.project_root(project_id)
        metadata = self.read_json_file(root / ".studio" / "project.json")
        if metadata:
            if metadata.get("project_type") == "movie":
                self.ensure_movie_config(root, metadata)
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
        atomic_write_json(path, metadata)

    @staticmethod
    def ensure_movie_config(root: Path, metadata: dict[str, Any]) -> None:
        path = root / "config.json"
        if path.exists():
            return
        atomic_write_json(path, movie_default_config_from_metadata(metadata))



def movie_default_config(request: ProjectCreateRequest) -> dict[str, Any]:
    return _movie_default_config(
        name=str(request.name).strip(),
        story_text=str(request.story_text or "").strip(),
        silent_mode=bool(request.silent_mode),
        width=int(request.width or 1280),
        height=int(request.height or 704),
        fps=int(request.fps or 24),
        dialogue_language=_dialogue_language(request.dialogue_language),
        render_profile=_render_profile_config(request),
    )


def movie_default_config_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    movie = dict(metadata.get("movie") or {})
    return _movie_default_config(
        name=str(metadata.get("display_name") or metadata.get("slug") or "").strip(),
        story_text=str(movie.get("story_text") or "").strip(),
        silent_mode=bool(metadata.get("silent_mode", False)),
        width=int(movie.get("width") or 1280),
        height=int(movie.get("height") or 704),
        fps=int(movie.get("fps") or 24),
        dialogue_language=_dialogue_language(movie.get("dialogue_language")),
        render_profile={"quality": "draft", "pass_strategy": "two_pass", "postprocess": "none"},
    )


def _movie_default_config(*, name: str, story_text: str, silent_mode: bool, width: int, height: int, fps: int, dialogue_language: str, render_profile: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "project_name": name,
        "content_mode": "narrative_film",
        "input_audio": "",
        "silent_mode": silent_mode,
        "lyrics": "",
        "dialogue_language": dialogue_language,
        "video": {
            "fps": fps,
            "width": width,
            "height": height,
        },
        "audio": {
            "demucs_model": "htdemucs_6s",
            "whisper_model": "large-v3",
            "language": "en",
        },
        "video_pipeline": "ltx_msr",
        "render_profile": dict(render_profile or {"quality": "draft", "pass_strategy": "two_pass", "postprocess": "none"}),
        "scene_generation": {
            "min_duration": 2.0,
            "max_duration": 10.0,
            "bias": 0.7,
            "duration_preset": "impact_weighted",
            "seed": -1,
        },
        "vocal_detection": {
            "merge_gap": 0.5,
            "min_vocal_duration": 0.4,
            "min_silence_duration": 0.8,
            "rms_low_percentile": 20,
            "rms_high_percentile": 85,
            "rms_ratio": 0.35,
            "smooth_frames": 10,
        },
        "story_idea": story_text,
        "style": "",
        "subject": "",
        "subject_mode": "multi",
        "max_scene_actors": 4,
        "locations": [],
        "actors": [],
        "steering": {
            "global": "",
            "story_idea": story_text,
            "style": "",
            "subject": "",
            "locations": "",
            "concepts": "",
            "zimage": "",
            "ltx": "",
            "final_prompts": "",
        },
        "prompt_guidance": {
            "character_visibility": "",
            "shot_types": "",
            "environments": "",
            "lighting": "",
            "camera_motion": "",
            "physical_interaction": "",
            "facial_expression": "",
            "outfit_rules": "",
            "prompt_structure": "",
            "list_handling": "",
            "word_count_min": SCENE_PROMPT_WORD_COUNT_MIN,
            "word_count_max": SCENE_PROMPT_WORD_COUNT_MAX,
        },
        "lora_1": {
            "enabled": False,
            "name": "",
            "strength_model": 1.0,
            "strength_clip": 1.0,
        },
        "lora_split_enabled": False,
        "loras": [],
    }
