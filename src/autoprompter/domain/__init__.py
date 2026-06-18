from __future__ import annotations

import importlib
import sys


_MODULES = [
    "ltx_rendering",
    "render_plan",
]

for _name in _MODULES:
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"domain.{_name}")

__all__ = list(_MODULES)
