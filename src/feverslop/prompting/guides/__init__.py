"""Bundled prompt-writing guides for the supported generation pipelines."""
from __future__ import annotations

from feverslop.prompting.guide_loader import load_markdown_guide


def load_guide(name: str) -> str:
    """Load a bundled Markdown guide by filename or stem."""
    return load_markdown_guide(name)
