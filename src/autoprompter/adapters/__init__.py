from __future__ import annotations

import importlib
import sys


_MODULES = [
    "comfyui_rendering",
    "comfyui_video_backend",
    "local_artifacts",
    "openai_compatible_llm",
]

for _name in _MODULES:
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"adapters.{_name}")

__all__ = list(_MODULES)
