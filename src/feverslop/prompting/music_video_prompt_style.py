from __future__ import annotations

from typing import Any

from feverslop.domain.prompt_constraints import build_location_constraint  # noqa: F401
from feverslop.prompting.guide_loader import load_markdown_guide


def performance_policy(segment_type: str, *, silent_mode: bool = False) -> str:
    if silent_mode:
        return (
            "Performance policy: dialogue-free silent mode is active. The subject must not sing, "
            "must have no lip sync, no vocal performance, no dialogue delivery, no mouth performance, "
            "and no moving lips. Preserve emotional acting through expressive eyes, facial emotion, "
            "gaze, posture, hands, body movement, camera movement, and environment."
        )
    mode = str(segment_type or "").strip().lower()
    if mode == "vocals":
        return "Performance policy: the subject is singing with passion and expressive lip sync, with the mouth performance matching the vocal energy and the face showing clear emotion."
    if mode == "mixed":
        return "Performance policy: the scene alternates between vocal intervals and silent intervals. Use singing with passion and lip sync only during vocal intervals. During silent intervals the subject must not sing, must have no lip sync, and keeps a closed or relaxed mouth."
    return "Performance policy: this is an instrumental section. The subject must not sing, must have no lip sync, no mouth performance, and no moving lips. Keep a closed or relaxed mouth while the emotion comes from gaze, posture, hands, body movement, camera movement, and environment."


def build_t2i_system_prompt() -> str:
    return load_markdown_guide("music-video-t2i")


def build_i2v_system_prompt(segment_type: str, *, silent_mode: bool = False) -> str:
    return "\n\n".join((
        load_markdown_guide("music-video-i2v"),
        performance_policy(segment_type, silent_mode=silent_mode),
    ))


def build_concept_mapper_system_prompt(*, batch: bool = False, silent_mode: bool = False) -> str:
    guide = load_markdown_guide("music-video-concepts")
    if batch:
        guide += "\n\nThis request contains one batch only; preserve continuity with prior progress."
    if silent_mode:
        guide += "\n\nSilent mode is active: do not create singing, lip-sync, vocal performance, mouth performance, or dialogue delivery."
    return guide


def build_detail_system_prompt(label: str, *, segment_type: str = "", silent_mode: bool = False) -> str:
    category_rules = {
        "camera motion": "For Camera Motion, output only camera movement phrases.",
        "character motion": "For Character Motion, output only visible body movement or performance movement.",
        "spatial relations": (
            "For Spatial Relations, describe only generic shot-level spatial facts: camera side or viewpoint, "
            "subject position and orientation, foreground/midground/background layer, relations between "
            "subjects and environment, visibility across the shot, and required prop bindings. "
            "Do not invent genre-specific staging rules."
        ),
        "lighting": "For Lighting, output only lighting descriptions.",
        "weather": "For Weather, output only weather descriptions.",
        "time of day": "For Time of Day, output only time-of-day phrases.",
        "emotion": f"For {label.strip()}, output only the emotion or expression.",
        "facial expression": f"For {label.strip()}, output only the emotion or expression.",
    }
    return "\n\n".join((
        load_markdown_guide("music-video-detail"),
        category_rules.get(label.strip().lower(), "Keep the line limited to the requested label."),
        performance_policy(segment_type, silent_mode=silent_mode),
    ))


def build_video_payload(*, segment: dict[str, Any], concept: str, scene_details: dict[str, Any], global_context: dict[str, Any], scene_cast: dict[str, Any] | None = None, t2i_prompt: str = "", custom_instructions: str = "") -> dict[str, Any]:
    segment_type = str(segment.get("type", "")).strip().lower()
    silent_mode = bool(global_context.get("silent_mode", False))
    return {
        "subject": global_context["subject"], "story_idea": global_context["story_idea"],
        "style": global_context["style"], "locations": global_context["locations"],
        "prompt_guidance": global_context.get("prompt_guidance", {}), "segment": segment,
        "performance_mode": segment_type, "silent_mode": silent_mode,
        "performance_policy": performance_policy(segment_type, silent_mode=silent_mode),
        "t2i_prompt": t2i_prompt, "scene_concept": concept, "scene_cast": scene_cast or {},
        "camera_motion": scene_details.get("camera_motion", ""),
        "character_motion": scene_details.get("character_motion", ""),
        "spatial_relations": scene_details.get("spatial_relations", ""),
        "custom_instructions": custom_instructions,
    }
