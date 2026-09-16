"""Shared stage progress reporting for the composition pipeline."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from feverslop.adapters.reporting import ConsoleReporter
from feverslop.ports.reporting import Reporter
from feverslop.utils.rich_progress import build_progress

VIDEO_SCENE_PROGRESS_LABEL = "Rendering video scenes"

console = Console()
_active_reporter: Reporter | None = None


def set_reporter(reporter: Reporter | None) -> None:
    global _active_reporter
    _active_reporter = reporter


def _report(text: str = "") -> None:
    (_active_reporter or ConsoleReporter(console)).message(text)


def report(text: str = "") -> None:
    _report(text)


class RenderProgressReporter:
    def __init__(
        self,
        description: str,
        total: int,
        *,
        console: Console = console,
        emit_scene_progress: bool = False,
    ):
        self.description = description
        self.total = total
        self.emit_scene_progress = emit_scene_progress
        self.progress = build_progress(console=console)
        self.task_id = None

    def __enter__(self) -> RenderProgressReporter:
        self.progress.__enter__()
        self.task_id = self.progress.add_task(self.description, total=self.total)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.progress.__exit__(exc_type, exc_value, traceback)

    def update(self, _output_path: Path, completed: int, total: int) -> None:
        if self.task_id is not None:
            self.progress.update(self.task_id, completed=completed)
        if self.emit_scene_progress:
            _report(f"Rendered scene {completed}/{total}")

    def analysis_attempt(self, scene_id: int, references: list[dict[str, str]]) -> None:
        summary = ", ".join(f"{item['type']}:{item['id']}" for item in references)
        _report(f"Ingredients image analysis: scene {scene_id}; {len(references)} references [{summary}]")
        if self.task_id is not None:
            self.progress.update(self.task_id, description=f"Analyzing scene {scene_id}: {summary}")
