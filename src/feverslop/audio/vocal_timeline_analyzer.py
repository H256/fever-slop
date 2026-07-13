from __future__ import annotations

from feverslop.adapters.audio.vocal_timeline_analyzer import VocalTimelineAnalyzer
from feverslop.domain.timeline import TimelineSegment
from feverslop.domain.timeline_transform import merge_same_kind_segments, normalize_empty_vocals

__all__ = [
    "TimelineSegment",
    "VocalTimelineAnalyzer",
    "merge_same_kind_segments",
    "normalize_empty_vocals",
]
