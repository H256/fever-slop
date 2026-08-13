from __future__ import annotations

from time import monotonic
from typing import Any


class SubStepProgress:
    """Throttle X/Y progress messages for loops inside a pipeline stage."""

    def __init__(self, reporter: Any, title: str, total: int, *, interval: int = 10, verbose: bool = False, quiet: bool = False) -> None:
        self.reporter = reporter
        self.title = str(title)
        self.total = max(0, int(total))
        self.interval = max(1, int(interval))
        self.verbose = bool(verbose)
        self.quiet = bool(quiet)
        self.started = monotonic()
        self._last = 0

    def update(self, current: int, *, detail: str = "", force: bool = False) -> None:
        current = max(0, min(int(current), self.total)) if self.total else max(0, int(current))
        if self.quiet or (not force and current != self.total and not self.verbose and current % self.interval):
            return
        if current == self._last and not force:
            return
        self._last = current
        elapsed = int(monotonic() - self.started)
        suffix = f" {detail.strip()}" if detail.strip() else ""
        self.reporter.message(f"[cyan]{self.title}: {current}/{self.total} [{elapsed // 60:02d}:{elapsed % 60:02d}]{suffix}[/cyan]")
