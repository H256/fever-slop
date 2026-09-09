"""Keep timing audit artifacts out of creative music-video model requests."""
from __future__ import annotations

from typing import Any


_EVIDENCE_FIELDS = frozenset({
    "performance_intervals", "word_timestamps", "performance_conflicts",
    "vocal_sources", "vocal_events", "alignment", "raw_words", "raw_text",
})


def compact_planning_payload(value: Any) -> Any:
    """Copy creative inputs without replicated acoustic evidence; never truncate lyrics.

    Deterministic timing and H3 compilation keep consuming the original artifacts.
    This projection is only for concepts, creative scene details and their repairs.
    """
    if isinstance(value, dict):
        return {key: compact_planning_payload(item) for key, item in value.items()
                if key not in _EVIDENCE_FIELDS}
    if isinstance(value, (list, tuple)):
        return [compact_planning_payload(item) for item in value]
    return value
