from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class TextArtifactReaderWriter(Protocol):
    def read_text(self, path: str | Path) -> str:
        """Read UTF-8 text from a path."""

    def write_text(self, path: str | Path, text: str) -> Path:
        """Write UTF-8 text and return the written path."""


class JsonArtifactStore(Protocol):
    def read_json(self, path: str | Path) -> Any:
        """Read JSON data from a path."""

    def write_json(self, path: str | Path, data: Any) -> Path:
        """Write JSON data and return the written path."""


class RenderPlanStore(Protocol):
    def read_render_plan(self, path: str | Path) -> list[dict]:
        """Read a render plan as the existing list-of-dicts contract."""

    def write_render_plan(self, path: str | Path, scenes: list[dict]) -> Path:
        """Write a render plan using the existing list-of-dicts contract."""


class ArtifactStore(TextArtifactReaderWriter, JsonArtifactStore, RenderPlanStore, Protocol):
    """Compatibility protocol for callers that still need all artifact capabilities."""
