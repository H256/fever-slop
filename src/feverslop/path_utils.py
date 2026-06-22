from __future__ import annotations

from os import PathLike
from pathlib import Path, PureWindowsPath
import os


def coerce_local_path(value: str | PathLike[str], *, base_dir: str | PathLike[str] | None = None) -> Path:
    """Coerce local path strings that may use Windows or POSIX separators."""
    raw = os.fspath(value)
    path = Path(raw)
    if _is_native_absolute(path) or _looks_like_windows_absolute(raw):
        coerced = path
    else:
        coerced = Path(raw.replace("\\", "/"))

    if base_dir is not None and not coerced.is_absolute():
        return Path(base_dir) / coerced
    return coerced


def _is_native_absolute(path: Path) -> bool:
    return path.is_absolute()


def _looks_like_windows_absolute(value: str) -> bool:
    windows_path = PureWindowsPath(value)
    return bool(windows_path.drive and windows_path.root)
