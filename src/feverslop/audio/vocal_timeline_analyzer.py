from __future__ import annotations

from feverslop.adapters.audio.vocal_timeline_analyzer import (
    TimelineSegment,
    VocalTimelineAnalyzer,
    merge_same_kind_segments,
    normalize_empty_vocals,
)

__all__ = [
    "TimelineSegment",
    "VocalTimelineAnalyzer",
    "merge_same_kind_segments",
    "normalize_empty_vocals",
]
