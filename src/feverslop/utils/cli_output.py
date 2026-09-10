"""Explicit machine-readable command-line output channel.

Human-facing status and error messages must use ``Reporter``.  Commands that
promise JSON or prompt text to scripts use this narrowly-scoped data channel so
their payloads stay parseable and are never decorated with timestamps.
"""

from __future__ import annotations

import sys
from typing import TextIO


def emit_cli_data(value: str, *, file: TextIO | None = None) -> None:
    """Write one complete machine-readable payload without Rich decoration."""
    stream = file or sys.stdout
    stream.write(value)
    if not value.endswith("\n"):
        stream.write("\n")
