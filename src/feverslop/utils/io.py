from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from feverslop.errors import FeverSlopDataError

_atomic_replace_lock = threading.Lock()


def _replace_atomically(source: Path, target: Path) -> None:
    with _atomic_replace_lock:
        os.replace(source, target)


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON document."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_json_document(path: str | Path) -> Any:
    """Read JSON and translate parse/I/O failures into a contextual data error."""
    candidate = Path(path)
    try:
        return read_json(candidate)
    except (FileNotFoundError, IsADirectoryError):
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise FeverSlopDataError(f"Cannot read JSON document: {candidate}: {exc}") from exc


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
            f"Cannot read movie pipeline artifact: {candidate}: {exc}",
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"Movie pipeline artifact must be a JSON object: {candidate}")
    return data


def atomic_write_json(path: Path, data: Any, **json_kwargs) -> Path:
    """Write JSON atomically: temp file in same dir, sync, then os.replace()."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    content = json.dumps(data, ensure_ascii=False, indent=2, **json_kwargs) + "\n"
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        _replace_atomically(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def write_json_document(path: str | Path, data: Any, **json_kwargs) -> Path:
    """Persist a JSON document through the shared atomic writer."""
    return atomic_write_json(Path(path), data, **json_kwargs)


def atomic_write_text(path: Path, text: str) -> Path:
    """Write text atomically: temp file in same dir, sync, then os.replace()."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        _replace_atomically(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def atomic_write_bytes(path: Path, data: bytes) -> Path:
    """Write binary data atomically: temp file in same dir, sync, then os.replace()."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _replace_atomically(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def file_is_valid(path: Path) -> bool:
    """Check that path exists as a file with non-zero size."""
    return path.is_file() and path.stat().st_size > 0
