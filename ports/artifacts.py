from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class ArtifactStore(Protocol):
    def read_json(self, path: str | Path) -> Any:
        """Read JSON data from a path."""

    def write_json(self, path: str | Path, data: Any) -> Path:
        """Write JSON data and return the written path."""

    def read_render_plan(self, path: str | Path) -> list[dict]:
        """Read a render plan as the existing list-of-dicts contract."""

    def write_render_plan(self, path: str | Path, scenes: list[dict]) -> Path:
        """Write a render plan using the existing list-of-dicts contract."""
