"""Jinja2 template loader for movie planning prompts."""
from __future__ import annotations

from pathlib import Path
from typing import Final
from jinja2 import Template

_PROMPTS_DIR: Final[Path] = Path(__file__).parent


def load_template(name: str) -> Template:
    """Load a .j2 template by base name from the prompts directory."""
    template_path = _PROMPTS_DIR / f"{name}.j2"
    source = template_path.read_text(encoding="utf-8")
    return Template(source)
