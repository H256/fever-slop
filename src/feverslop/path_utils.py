from __future__ import annotations

from os import PathLike
from pathlib import Path, PureWindowsPath
import os


def coerce_local_path(
    value: str | PathLike[str],
    *,
    base_dir: str | PathLike[str] | None = None,
    containment_root: str | PathLike[str] | None = None,
) -> Path:
    """Coerce local path strings that may use Windows or POSIX separators.

    When *containment_root* is provided the resolved path must stay under
    that root directory.  Raises :class:`ValueError` on escape attempts
    (e.g. ``../`` traversal).
    """
    raw = os.fspath(value)
    path = Path(raw)
    if _is_native_absolute(path) or _looks_like_windows_absolute(raw):
        coerced = path
    else:
        coerced = Path(raw.replace("\\", "/"))

    if base_dir is not None and not coerced.is_absolute():
        coerced = Path(base_dir) / coerced

    if containment_root is not None:
        resolved = coerced.resolve()
        root = Path(containment_root).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(
                f"Path escapes containment root: {resolved} is not under {root}"
            )
    return coerced


def _is_native_absolute(path: Path) -> bool:
    return path.is_absolute()


def _looks_like_windows_absolute(value: str) -> bool:
    windows_path = PureWindowsPath(value)
    return bool(windows_path.drive and windows_path.root)
