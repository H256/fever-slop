from __future__ import annotations

from feverslop.adapters.audio.vocal_timeline_analyzer import (
    VocalTimelineAnalyzer,
    merge_same_kind_segments,
    normalize_empty_vocals,
)
from feverslop.domain.timeline import TimelineSegment

__all__ = [
    "TimelineSegment",
    "VocalTimelineAnalyzer",
    "merge_same_kind_segments",
    "normalize_empty_vocals",
]
