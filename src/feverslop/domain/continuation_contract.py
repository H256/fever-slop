from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Iterable

from feverslop.domain.continuation_segments import SemanticRenderSegment


@dataclass(frozen=True)
class ContinuationGroup:
    group_id: str
    semantic_action: str
    semantic_start_seconds: float
    semantic_end_seconds: float
    segments: tuple[SemanticRenderSegment, ...]

    @classmethod
    def create(
        cls, *, group_id: str, semantic_action: str,
        semantic_start_seconds: float, semantic_end_seconds: float,
        segments: Iterable[SemanticRenderSegment],
    ) -> "ContinuationGroup":
        identifier = str(group_id).strip()
        action = str(semantic_action).strip()
        start = float(semantic_start_seconds)
        end = float(semantic_end_seconds)
        normalized = tuple(segments)
        if not identifier or not action or end <= start:
            raise ValueError("continuation group requires a valid ID, action, and semantic interval")
        if not normalized:
            raise ValueError("continuation group requires at least one segment")
        expected_start = start
        expected_index = 1
        seen_ids: set[str] = set()
        for segment in normalized:
            if not isinstance(segment, SemanticRenderSegment):
                raise TypeError("segments must contain SemanticRenderSegment values")
            if segment.index != expected_index or segment.segment_id in seen_ids:
                raise ValueError("continuation segment indexes and IDs must be unique and contiguous")
            if not isclose(segment.start_seconds, expected_start, abs_tol=1e-6):
                raise ValueError("continuation segments contain a gap or overlap")
            if segment.end_seconds <= segment.start_seconds:
                raise ValueError("continuation segment interval must be positive")
            if segment.starts_with_anchor != (expected_index > 1):
                raise ValueError("only continuation segments after the first may start with an anchor")
            seen_ids.add(segment.segment_id)
            expected_start = segment.end_seconds
            expected_index += 1
        if not isclose(expected_start, end, abs_tol=1e-6):
            raise ValueError("segment coverage does not match semantic interval")
        return cls(identifier, action, start, end, normalized)

    def predecessor(self, segment_id: str) -> str | None:
        for index, segment in enumerate(self.segments):
            if segment.segment_id == str(segment_id).strip():
                return None if index == 0 else self.segments[index - 1].segment_id
        raise KeyError(segment_id)

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "semantic_action": self.semantic_action,
            "semantic_start_seconds": self.semantic_start_seconds,
            "semantic_end_seconds": self.semantic_end_seconds,
            "segments": [segment.__dict__.copy() for segment in self.segments],
        }
