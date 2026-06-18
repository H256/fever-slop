from __future__ import annotations

import importlib
import sys


_MODULES = [
    "artifacts",
    "audio",
    "llm",
    "postprocessing",
    "rendering",
    "workflow",
]

for _name in _MODULES:
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"ports.{_name}")

__all__ = list(_MODULES)
