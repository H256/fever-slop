from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypeVar

from rich.console import Console
from rich.markup import escape

T = TypeVar("T")


class Reporter(Protocol):
    def step(self, title: str) -> None:
        """Report a major step."""

    def file(self, label: str, path: Path) -> None:
        """Report a generated file."""

    def message(self, text: str) -> None:
        """Report a plain status message."""

    def warning(self, text: str, *, title: str | None = None) -> None:
        """Report a warning with adapter-owned Rich styling."""

    def panel(self, text: str, *, title: str | None = None) -> None:
        """Report prominent text."""

    def table(self, title: str, columns: list[str], rows: list[list[str]]) -> None:
        """Report tabular data."""

    def run_progress(self, description: str, func: Callable[[], T]) -> T:
        """Run work with optional progress reporting."""


class NullReporter:
    """No-op reporter used by application services without a UI."""

    def step(self, title: str) -> None:
        pass
    def file(self, label: str, path: Path) -> None:
        pass
    def message(self, text: str) -> None:
        pass
    def warning(self, text: str, *, title: str | None = None) -> None:
        pass
    def panel(self, text: str, *, title: str | None = None) -> None:
        pass
    def table(self, title: str, columns: list[str], rows: list[list[str]]) -> None:
        pass
    def run_progress(self, description: str, func: Callable[[], T]) -> T:
        return func()


class ConsoleReporter:
    """Rich-backed reporter assembled by composition roots."""

    def __init__(self, console: Console):
        self.console = console
    def step(self, title: str) -> None:
        self.console.print()
        self.console.rule(f"[bold cyan]{title}[/bold cyan]")
    def file(self, label: str, path: Path) -> None:
        self.message(f"[green]OK[/green] {label}: [cyan]{path}[/cyan]")
    def message(self, text: str) -> None:
        self.console.print(text)
    def warning(self, text: str, *, title: str | None = None) -> None:
        self.panel(f"[yellow]{escape(text)}[/yellow]", title=f"[bold yellow]{escape(title or 'Warning')}[/bold yellow]")
    def panel(self, text: str, *, title: str | None = None) -> None:
        heading = f"{title}\n" if title else ""
        self.message(f"{heading}{text}")
    def table(self, title: str, columns: list[str], rows: list[list[str]]) -> None:
        self.message(title)
        self.message(" | ".join(columns))
        for row in rows:
            self.message(" | ".join(row))
    def run_progress(self, description: str, func: Callable[[], T]) -> T:
        return func()
