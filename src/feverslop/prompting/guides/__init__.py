"""Bundled prompt-writing guides for the supported generation pipelines."""
from __future__ import annotations

from pathlib import Path

_GUIDES_DIR = Path(__file__).parent


def load_guide(name: str) -> str:
    """Load a bundled Markdown guide by filename or stem."""
    filename = name if name.endswith(".md") else f"{name}.md"
    path = _GUIDES_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Prompt guide not found: {path}")
    return path.read_text(encoding="utf-8")