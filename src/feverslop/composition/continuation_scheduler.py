from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from feverslop.domain.continuation_dependencies import ContinuationDependencyGraph
from feverslop.utils.sub_step_progress import SubStepProgress


def chains_from_predecessors(
    scene_numbers: list[int] | tuple[int, ...],
    predecessors: Mapping[int, int],
) -> dict[str, tuple[str, ...]]:
    """Build deterministic scheduler chains from the render-path handoffs."""
    ordered = tuple(sorted({int(number) for number in scene_numbers}))
    selected = set(ordered)
    successors: dict[int, int] = {}
    for successor, predecessor in predecessors.items():
        successor_number = int(successor)
        predecessor_number = int(predecessor)
        if successor_number not in selected or predecessor_number not in selected:
            continue
        if predecessor_number in successors:
            raise ValueError(f"scene {predecessor_number} has multiple continuation successors")
        successors[predecessor_number] = successor_number

    chains: dict[str, tuple[str, ...]] = {}
    visited: set[int] = set()
    for first in ordered:
        if first in visited or first in predecessors:
            continue
        chain: list[str] = []
        current = first
        while current in selected and current not in visited:
            visited.add(current)
            chain.append(f"scene-{current}")
            current = successors.get(current, -1)
        chains[chain[0]] = tuple(chain)
    for number in ordered:
        if number not in visited:
            chains[f"scene-{number}"] = (f"scene-{number}",)
    return chains


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
