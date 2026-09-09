from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_debug_workflow(path: Path, workflow: dict[str, Any], *, trailing_newline: bool = False) -> None:
    """Persist a resolved workflow for debugging, creating its directory first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(workflow, ensure_ascii=False, indent=2)
    if trailing_newline:
        serialized += "\n"
    path.write_text(serialized, encoding="utf-8")
