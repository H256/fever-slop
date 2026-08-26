"""Transport-neutral project persistence contracts shared by adapters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


class StudioPathError(ValueError):
    pass


class ArtifactConflict(ValueError):
    def __init__(self, path: str, expected_revision: str | None, actual_revision: str | None):
        message = (
            f"Artifact already exists; load target before saving: {path}"
            if expected_revision is None and actual_revision is not None
            else f"Artifact changed: {path}"
        )
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
    render_quality: str = "draft"
    render_pass_strategy: str = "two_pass"
    render_postprocess: str = "none"


AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
AUDIO_MIME_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/flac",
    "audio/x-flac", "audio/mp4", "audio/x-m4a", "audio/ogg",
}


def sanitize_audio_filename(value: str) -> str:
    name = re.split(r"[\\/]+", str(value or "").strip())[-1]
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not name:
        raise ValueError("Audio filename is required")
    return name
