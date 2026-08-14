from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any


GUIDE_PACKAGE = "feverslop.prompting.guides"


class PromptGuideNotFoundError(FileNotFoundError):
    """Raised when a bundled Markdown prompt guide cannot be resolved."""


def _guide_filename(name: str | Path) -> str:
    filename = Path(str(name)).name
    if filename and not filename.endswith(".md"):
        filename = f"{filename}.md"
    return filename


def resolve_guide_path(name: str | Path) -> Any:
    """Resolve a guide filename or legacy path to the package-local guide."""
    filename = _guide_filename(name)
    guide = files(GUIDE_PACKAGE).joinpath(filename)
    if not filename or not guide.is_file():
        raise PromptGuideNotFoundError(f"Prompt guide not found: {filename or name}")
    return guide


def load_markdown_guide(name: str | Path) -> str:
    """Load a bundled Markdown guide as UTF-8 text."""
    return resolve_guide_path(name).read_text(encoding="utf-8")
