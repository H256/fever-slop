from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from feverslop.errors import FeverSlopDataError


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON document."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_json_or_none(path: str | Path) -> Any | None:
    """Read a JSON document, returning None only when it is absent."""
    candidate = Path(path)
    if not candidate.exists():
        return None
    return read_json(candidate)


def read_json_object(path: str | Path) -> dict[str, Any]:
    """Read a JSON object used as a pipeline artifact.

    Missing files and directories are deliberately allowed to propagate so
    callers can keep their existing EAFP fallbacks. Parse and I/O failures
    get the same domain error, while a valid non-object JSON document remains
    a validation error.
    """
    candidate = Path(path)
    try:
        data = read_json(candidate)
    except (FileNotFoundError, IsADirectoryError):
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise FeverSlopDataError(
            f"Cannot read movie pipeline artifact: {candidate}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"Movie pipeline artifact must be a JSON object: {candidate}")
    return data


def atomic_write_json(path: Path, data: Any, **json_kwargs) -> Path:
    """Write JSON atomically: temp file in same dir, sync, then os.replace()."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    content = json.dumps(data, ensure_ascii=False, indent=2, **json_kwargs) + "\n"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def atomic_write_text(path: Path, text: str) -> Path:
    """Write text atomically: temp file in same dir, sync, then os.replace()."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def file_is_valid(path: Path) -> bool:
    """Check that path exists as a file with non-zero size."""
    return path.is_file() and path.stat().st_size > 0
