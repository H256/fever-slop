from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from feverslop.domain.continuation_dependencies import ContinuationDependencyGraph
from feverslop.utils.sub_step_progress import SubStepProgress


class ContinuationScheduler:
    """Run continuation segments in dependency order with observable progress."""

    def __init__(self, chains: Mapping[str, Iterable[str]], *, reporter: Any = None) -> None:
        self.graph = ContinuationDependencyGraph.from_chains(dict(chains))
        self.reporter = reporter

    def run(self, render_segment: Callable[[str], bool]) -> tuple[str, ...]:
        """Render ready segments until all chains complete or a boundary is invalid."""
        total = sum(len(chain) for chain in self.graph.chains)
        progress = SubStepProgress(self.reporter, "Continuation segments", total, interval=1)
        completed: list[str] = []
        blocked: set[str] = set()
        while ready := tuple(segment for segment in self.graph.ready() if segment not in blocked):
            segment_id = ready[0]
            self._message(f"Continuation segment started: {segment_id}")
            boundary_valid = bool(render_segment(segment_id))
            if not boundary_valid:
                self._message(f"Continuation boundary invalid: {segment_id}")
                blocked.add(segment_id)
                progress.update(
                    len(completed) + len(blocked),
                    detail=f"blocked={segment_id}",
                    force=True,
                )
                continue
            self.graph.mark_complete(segment_id, anchor_valid=True)
            completed.append(segment_id)
            progress.update(len(completed), detail=f"segment={segment_id}", force=True)
            self._message(f"Continuation boundary verified: {segment_id}")
        return tuple(completed)

    def _message(self, text: str) -> None:
        if self.reporter is not None:
            self.reporter.message(text)
