from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from feverslop.ports.movie import MovieArtifactWriter


class LocalMovieArtifactWriter(MovieArtifactWriter):
    """Concrete MovieArtifactWriter backed by the local filesystem."""

    def write_json(self, path: str | Path, data: Any) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def write_text(self, path: str | Path, text: str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def ensure_dir(self, path: str | Path, *, exist_ok: bool = True) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=exist_ok)
        return path
