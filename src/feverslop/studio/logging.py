"""Compatibility import for canonical headless log rendering."""

from feverslop.composition import logging as _canonical_logging
from feverslop.composition.logging import render_log_lines

_logger = _canonical_logging._logger

__all__ = ["render_log_lines"]
