from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from feverslop.adapters.movie_planning import DeterministicMoviePlanner, LLMMoviePlanner
from feverslop.application.movie import MovieInput, ScaffoldMovieUseCase
from feverslop.ports.reporting import Reporter
from feverslop.studio.project_validation import validate_full_auto_inputs
from feverslop.studio.projects import ProjectCreateRequest, StudioPathError, slugify_project_name


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
        movie_config = movie_project_config(request)
        config = movie_default_config(request)
        result = ScaffoldMovieUseCase(
            planner=build_movie_planner(movie_config),
            projects_root=self.projects_root,
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
                config=config,
            )
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
        path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @staticmethod
    def ensure_movie_config(root: Path, metadata: dict[str, Any]) -> None:
        path = root / "config.json"
        if path.exists():
            return
        path.write_text(json.dumps(movie_default_config_from_metadata(metadata), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def movie_project_config(request: ProjectCreateRequest) -> dict[str, Any]:
    planner_backend = _movie_planner_backend(request.movie_planner_backend)
    reference_backend = _supported_backend(request.movie_reference_backend, "movie_reference_backend", {"comfyui", "local"}, default="comfyui")
    render_backend = _supported_backend(request.movie_render_backend, "movie_render_backend", {"comfyui", "local"}, default="comfyui")
    movie_video_workflow = _supported_backend(request.movie_video_workflow, "movie_video_workflow", {"msr", "msr-i2v-startframe", "i2v-edit", "startframe-director"}, default="msr")
    startframe_director_backend = _supported_backend(
        request.movie_startframe_director_backend,
        "movie_startframe_director_backend",
        {"krea2", "ideogram"},
        default="krea2",
    )
    continuity_keyframes = _supported_backend(request.movie_continuity_keyframes, "movie_continuity_keyframes", {"none", "last-to-start"}, default="none")
    if continuity_keyframes == "last-to-start" and movie_video_workflow != "msr-i2v-startframe":
        raise ValueError("movie_continuity_keyframes=last-to-start requires movie_video_workflow=msr-i2v-startframe")
    edit_workflow = _movie_edit_workflow(request.movie_edit_workflow, movie_video_workflow=movie_video_workflow)
    return {
        "planner_backend": planner_backend,
        "reference_backend": reference_backend,
        "render_backend": render_backend,
        "movie_video_workflow": movie_video_workflow,
        "startframe_director_backend": startframe_director_backend,
        "continuity_keyframes": continuity_keyframes,
        "dialogue_language": _dialogue_language(request.dialogue_language),
        "hero_workflow": _project_workflow_path(request.movie_hero_workflow, "movie_hero_workflow"),
        "edit_workflow": edit_workflow,
        "director_workflow": _project_workflow_path(
            _default_movie_director_workflow(request.movie_director_workflow, startframe_director_backend),
            "movie_director_workflow",
        ),
        "mask_workflow": _project_workflow_path(request.movie_mask_workflow, "movie_mask_workflow"),
        "identity_repair_workflow": _project_workflow_path(request.movie_identity_repair_workflow, "movie_identity_repair_workflow"),
        "detail_workflow": _project_workflow_path(request.movie_detail_workflow, "movie_detail_workflow"),
        "startframe_comfyui_base_url": str(request.movie_startframe_comfyui_base_url or "http://localhost:8188").rstrip("/"),
        "startframe_write_debug_workflows": bool(request.movie_startframe_write_debug_workflows),
        "startframe_debug_workflows_dir": _project_optional_relative_path(
            request.movie_startframe_debug_workflows_dir,
            "movie_startframe_debug_workflows_dir",
        ),
        "startframe_validator_base_url": str(request.movie_startframe_validator_base_url or "http://llm.elysium.lan/v1").rstrip("/"),
        "startframe_validator_model": str(request.movie_startframe_validator_model or "gemma4-26b-a4b:vision"),
        "msr_workflow": _project_workflow_path(request.movie_msr_workflow, "movie_msr_workflow"),
        "msr_i2v_workflow": _project_workflow_path(request.movie_msr_i2v_workflow, "movie_msr_i2v_workflow"),
        "i2v_workflow": _project_workflow_path(request.movie_i2v_workflow, "movie_i2v_workflow"),
    }


def movie_default_config(request: ProjectCreateRequest) -> dict[str, Any]:
    return _movie_default_config(
        name=str(request.name).strip(),
        story_text=str(request.story_text or "").strip(),
        silent_mode=bool(request.silent_mode),
        width=int(request.width or 1280),
        height=int(request.height or 704),
        fps=int(request.fps or 24),
        dialogue_language=_dialogue_language(request.dialogue_language),
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
    )


def _movie_default_config(*, name: str, story_text: str, silent_mode: bool, width: int, height: int, fps: int, dialogue_language: str) -> dict[str, Any]:
    return {
        "project_name": name,
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
            "demucs_model": "htdemucs_ft",
            "whisper_model": "large",
            "language": "en",
        },
        "video_pipeline": "ltx_msr",
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
            "word_count_min": 40,
            "word_count_max": 50,
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


def build_movie_planner(config: dict[str, Any] | None = None):
    backend = str((config or {}).get("planner_backend") or "llm").strip().lower()
    if backend in {"deterministic", "local", "placeholder"}:
        return DeterministicMoviePlanner()
    if backend != "llm":
        raise ValueError("movie_planner_backend must be llm or deterministic")

    from feverslop.adapters.openai_compatible_llm import OpenAICompatibleLLMClient
    from feverslop.config.app_config import AppConfig

    app_config = AppConfig.load("app_config.json")
    return LLMMoviePlanner(
        OpenAICompatibleLLMClient(
            base_url=app_config.llm.base_url,
            model=app_config.llm.model,
            temperature=app_config.llm.temperature,
            max_tokens=app_config.llm.max_tokens,
        )
    )


def _movie_planner_backend(value: str) -> str:
    normalized = str(value or "llm").strip().lower()
    if normalized in {"local", "placeholder"}:
        normalized = "deterministic"
    if normalized not in {"llm", "deterministic"}:
        raise ValueError("movie_planner_backend must be llm or deterministic")
    return normalized


def _supported_backend(value: str, field: str, supported: set[str], *, default: str) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        normalized = default
    if normalized == "placeholder":
        normalized = "local"
    if normalized not in supported:
        raise ValueError(f"{field} must be one of: {', '.join(sorted(supported))}")
    return normalized


def _project_workflow_path(value: str, field: str) -> str:
    path = str(value or "").strip()
    if not path:
        raise ValueError(f"{field} is required")
    parsed = Path(path)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise ValueError(f"{field} must be a repository-relative path")
    return parsed.as_posix()


def _project_optional_relative_path(value: str, field: str) -> str:
    path = str(value or "").strip()
    if not path:
        return ""
    parsed = Path(path)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise ValueError(f"{field} must be a repository-relative path")
    return parsed.as_posix()


def _movie_edit_workflow(value: str, *, movie_video_workflow: str) -> str:
    default_1ref = "workflows/image_edit_flux2_klein_1ref_v1.json"
    if movie_video_workflow == "i2v-edit" and str(value or "").strip() in {"", default_1ref}:
        return "workflows/image_edit_flux2_klein_2ref_v1.json"
    return _project_workflow_path(value, "movie_edit_workflow")


def _default_movie_director_workflow(value: str, backend: str) -> str:
    raw = str(value or "").strip()
    krea_default = "workflows/image_t2i_startframe_krea_v1.json"
    if raw:
        if backend == "ideogram" and raw == krea_default:
            return "workflows/image_t2i_startframe_ideogram_director_v1.json"
        return raw
    if backend == "ideogram":
        return "workflows/image_t2i_startframe_ideogram_director_v1.json"
    return krea_default


def _dialogue_language(value: object) -> str:
    language = str(value or "").strip()
    return language or "English"
