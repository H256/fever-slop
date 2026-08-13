from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol, TypeVar

T = TypeVar("T")


class Reporter(Protocol):
    def step(self, title: str) -> None:
        """Report a major step."""

    def file(self, label: str, path: Path) -> None:
        """Report a generated file."""

    def message(self, text: str) -> None:
        """Report a plain status message."""

    def panel(self, text: str, *, title: str | None = None) -> None:
        """Report prominent text."""

    def table(self, title: str, columns: list[str], rows: list[list[str]]) -> None:
        """Report tabular data."""

    def run_progress(self, description: str, func: Callable[[], T]) -> T:
        """Run work with optional progress reporting."""
