from __future__ import annotations

from typing import Any

from feverslop.application.movie import (
    _dialogue_language,
    _render_profile_config,
)
from feverslop.composition.movie_planner import build_movie_planner  # noqa: F401
from feverslop.config.project_config import (
    SCENE_PROMPT_WORD_COUNT_MAX,
    SCENE_PROMPT_WORD_COUNT_MIN,
)
from feverslop.ports.project_requests import ProjectCreateRequest


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
