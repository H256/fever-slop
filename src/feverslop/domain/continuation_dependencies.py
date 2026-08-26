from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ContinuationDependencyGraph:
    """Small deterministic dependency graph for serial continuation chains."""

    chains: tuple[tuple[str, ...], ...]
    _states: dict[str, str]

    @classmethod
    def from_chains(cls, chains: dict[str, Iterable[str]]) -> "ContinuationDependencyGraph":
        normalized = tuple(
            tuple(str(segment).strip() for segment in segments)
            for _, segments in sorted(chains.items())
        )
        if any(not chain or any(not segment for segment in chain) for chain in normalized):
            raise ValueError("continuation chains must contain non-blank segment IDs")
        ids = [segment for chain in normalized for segment in chain]
        if len(ids) != len(set(ids)):
            raise ValueError("continuation segment IDs must be unique")
        return cls(normalized, {segment: "pending" for segment in ids})

    def ready(self) -> tuple[str, ...]:
        result: list[str] = []
        for chain in self.chains:
            for index, segment in enumerate(chain):
                if self._states[segment] != "pending":
                    continue
                if index == 0 or self._states[chain[index - 1]] == "complete":
                    result.append(segment)
                break
        return tuple(sorted(result))

    def mark_complete(self, segment_id: str, *, anchor_valid: bool) -> None:
        segment = str(segment_id).strip()
        if segment not in self._states:
            raise KeyError(segment)
        if segment not in self.ready():
            raise ValueError(f"segment is not ready: {segment}")
        if not anchor_valid:
            raise ValueError(f"verified boundary anchor is required: {segment}")
        self._states[segment] = "complete"

    def invalidate_suffix(self, segment_id: str) -> tuple[str, ...]:
        segment = str(segment_id).strip()
        for chain in self.chains:
            if segment not in chain:
                continue
            start = chain.index(segment)
            affected = chain[start:]
            for item in affected:
                self._states[item] = "pending"
            return tuple(affected)
        raise KeyError(segment)
