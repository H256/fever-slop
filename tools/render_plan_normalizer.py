"""Legacy facade for packaged render-plan normalization helpers."""

from feverslop.tools.render_plan_normalizer import (
    frame_count_from_duration,
    normalize_render_plan,
    normalize_render_plan_file,
    scene_duration_from_frame_count,
)

__all__ = [
    "frame_count_from_duration",
    "normalize_render_plan",
    "normalize_render_plan_file",
    "scene_duration_from_frame_count",
]
