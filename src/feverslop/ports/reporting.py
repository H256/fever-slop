from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Protocol, TextIO, TypeVar

from rich.console import Console
from rich.markup import escape

T = TypeVar("T")

TRACE_LEVEL = 5
OFF_LEVEL = logging.CRITICAL + 1
_LOG_LEVELS = {
    "trace": TRACE_LEVEL,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "off": OFF_LEVEL,
}
logging.addLevelName(TRACE_LEVEL, "TRACE")


def parse_log_level(value: str | int | None) -> int:
    """Resolve the global human-facing log level; default to INFO."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return logging.INFO
    if isinstance(value, int):
        if value < 0:
            raise ValueError("log level must not be negative")
        return value
    normalized = value.strip().lower()
    try:
        return _LOG_LEVELS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unknown log level {value!r}; expected trace, debug, info, "
            "warning, error, critical, or off",
        ) from exc


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
        self.console.rule(self._timestamp(f"[bold cyan]{title}[/bold cyan]"))
    def file(self, label: str, path: Path) -> None:
        self.message(f"[green]OK[/green] {label}: [cyan]{path}[/cyan]")
    def message(self, text: str) -> None:
        self.console.print(self._timestamp(text))

    @staticmethod
    def _timestamp(text: str) -> str:
        now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        return "\n".join(f"[dim][{now}][/dim] {line}" for line in text.splitlines())
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


class ReporterConsole:
    """Compatibility console that routes human output through a Reporter."""

    def __init__(self, console: Console, reporter: ConsoleReporter | None = None):
        self.console = console
        self.reporter = reporter or ConsoleReporter(console)

    def print(self, *objects: object, **kwargs: object) -> None:
        if not objects:
            self.reporter.message("")
            return
        if all(isinstance(value, str) for value in objects) and not kwargs:
            self.reporter.message(" ".join(objects))
            return
        timestamp = self.reporter._timestamp(" ")
        self.console.print(timestamp, *objects, **kwargs)

    def __getattr__(self, name: str) -> object:
        return getattr(self.console, name)


def report_message(*objects: object, file: TextIO | None = None, **_kwargs: object) -> None:
    """Report a short CLI status line through a timestamped Rich reporter.

    This compatibility helper is for small command-line tools that do not yet
    have a composed application reporter.  It resolves the stream at call time
    so tests and embedding callers can redirect stdout/stderr reliably.
    """
    ConsoleReporter(Console(file=file)).message(" ".join(str(value) for value in objects))


class ReporterLoggingHandler(logging.Handler):
    """Forward application and captured warning logs through the Reporter."""

    def __init__(self, reporter: Reporter):
        super().__init__()
        self.reporter = reporter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            if record.levelno >= logging.WARNING:
                self.reporter.warning(message, title=record.name)
            else:
                self.reporter.message(f"[dim]{record.name}[/dim] {message}")
        except Exception:
            self.handleError(record)


def install_reporter_logging(reporter: Reporter, *, level: str | int | None = None) -> None:
    resolved_level = parse_log_level(
        level if level is not None else os.environ.get("FEVERSLOP_LOG_LEVEL"),
    )
    root = logging.getLogger()
    handler = next(
        (item for item in root.handlers if isinstance(item, ReporterLoggingHandler)),
        None,
    )
    if handler is None:
        handler = ReporterLoggingHandler(reporter)
        root.addHandler(handler)
    else:
        handler.reporter = reporter
    handler.setLevel(resolved_level)
    root.setLevel(resolved_level)
    for name in ("litellm", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(resolved_level)
    logging.captureWarnings(True)
