from __future__ import annotations

import io
import logging
import re
from typing import Any

from rich.console import Console

_logger = logging.getLogger(__name__)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
RICH_TAG_RE = re.compile(r"\[/?[a-zA-Z][a-zA-Z0-9_ #=;.,:'\"-]*\]")


def render_log_lines(*values: Any) -> list[str]:
    """Render Rich renderables/markup to plain text lines for Studio logs."""
    if not values:
        return []
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, width=120, markup=True, highlight=False)
    for value in values:
        try:
            console.print(value)
        except ValueError as e:
            _logger.debug("render_log_lines ValueError for %r: %s", value, e, exc_info=True)
            console.print(str(value))
        except RuntimeError as e:
            _logger.debug("render_log_lines RuntimeError for %r: %s", value, e, exc_info=True)
            console.print(str(value))
    text = ANSI_RE.sub("", buffer.getvalue())
    text = RICH_TAG_RE.sub("", text)
    return [line.rstrip() for line in text.splitlines() if line.strip()]
