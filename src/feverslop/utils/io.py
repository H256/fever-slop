from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON document."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_json_or_none(path: str | Path) -> Any | None:
    """Read a JSON document, returning None only when it is absent."""
    candidate = Path(path)
    if not candidate.exists():
        return None
    return read_json(candidate)
