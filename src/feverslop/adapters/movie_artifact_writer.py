from __future__ import annotations

from pathlib import Path
from typing import Any

from feverslop.ports.movie import MovieArtifactWriter
from feverslop.utils.io import atomic_write_json, atomic_write_text


class LocalMovieArtifactWriter(MovieArtifactWriter):
    """Concrete MovieArtifactWriter backed by the local filesystem."""

    def write_json(self, path: str | Path, data: Any) -> Path:
        return atomic_write_json(Path(path), data)

    def write_text(self, path: str | Path, text: str) -> Path:
        return atomic_write_text(Path(path), text)

    def ensure_dir(self, path: str | Path, *, exist_ok: bool = True) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=exist_ok)
        return path
