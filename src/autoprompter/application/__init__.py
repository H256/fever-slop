from __future__ import annotations

import importlib
import sys


_MODULES = [
    "audio_timeline_pipeline",
    "generate_render_plan",
    "llm_parsing",
    "prompt_generation_pipeline",
    "render_plan_pipeline",
    "render_storyboard",
    "render_video",
    "scene_timeline_pipeline",
]

for _name in _MODULES:
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"application.{_name}")

__all__ = list(_MODULES)
