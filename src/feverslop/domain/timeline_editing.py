from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EditableTimelineSegment:
    """A TimelineSegment enriched with editing metadata."""

    start: float
    end: float
    kind: str
    text: str = ""
    lyrics_line: str | None = None
    notes: str | None = None
    is_draft: bool = False

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"start must be >= 0, got {self.start}")
        if self.end < 0:
            raise ValueError(f"end must be >= 0, got {self.end}")
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")


@dataclass(frozen=True)
class SceneBoundary:
    """Marks the start/end of a scene with validation on minimum duration."""

    start: float
    end: float
    reason: str
    min_duration: float = 2.0

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"start must be >= 0, got {self.start}")
        if self.end < 0:
            raise ValueError(f"end must be >= 0, got {self.end}")
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")
        duration = self.end - self.start
        if duration < self.min_duration:
            raise ValueError(
                f"duration ({duration}) must be >= min_duration ({self.min_duration})"
            )


@dataclass(frozen=True)
class BeatMarker:
    """A detected beat position with label and confidence."""

    time_s: float
    label: str
    confidence: float

    def __post_init__(self) -> None:
        if self.time_s < 0:
            raise ValueError(f"time_s must be >= 0, got {self.time_s}")
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")


@dataclass(frozen=True)
class TimelineSnapshot:
    """Immutable snapshot of the editable timeline state."""

    segments: list[EditableTimelineSegment] | tuple[EditableTimelineSegment, ...]
    scene_boundaries: list[SceneBoundary] | tuple[SceneBoundary, ...]
    beat_markers: list[BeatMarker] | tuple[BeatMarker, ...]
    metadata: dict

    def __post_init__(self) -> None:
        # Normalize mutable inputs to immutable representations.
        object.__setattr__(self, "segments", tuple(self.segments))
        object.__setattr__(self, "scene_boundaries", tuple(self.scene_boundaries))
        object.__setattr__(self, "beat_markers", tuple(self.beat_markers))
        # Wrap metadata as a read-only proxy.
        import types

        object.__setattr__(
            self, "metadata", types.MappingProxyType(dict(self.metadata))
        )

    def to_json(self) -> dict:
        return {
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "kind": s.kind,
                    "text": s.text,
                    "lyrics_line": s.lyrics_line,
                    "notes": s.notes,
                    "is_draft": s.is_draft,
                }
                for s in self.segments
            ],
            "scene_boundaries": [
                {
                    "start": b.start,
                    "end": b.end,
                    "reason": b.reason,
                    "min_duration": b.min_duration,
                }
                for b in self.scene_boundaries
            ],
            "beat_markers": [
                {
                    "time_s": m.time_s,
                    "label": m.label,
                    "confidence": m.confidence,
                }
                for m in self.beat_markers
            ],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_json(cls, data: dict) -> TimelineSnapshot:
        segments = tuple(
            EditableTimelineSegment(
                start=s["start"],
                end=s["end"],
                kind=s["kind"],
                text=s.get("text", ""),
                lyrics_line=s.get("lyrics_line"),
                notes=s.get("notes"),
                is_draft=s.get("is_draft", False),
            )
            for s in data["segments"]
        )
        boundaries = tuple(
            SceneBoundary(
                start=b["start"],
                end=b["end"],
                reason=b["reason"],
                min_duration=b.get("min_duration", 2.0),
            )
            for b in data["scene_boundaries"]
        )
        markers = tuple(
            BeatMarker(
                time_s=m["time_s"],
                label=m["label"],
                confidence=m["confidence"],
            )
            for m in data["beat_markers"]
        )
        return cls(
            segments=segments,
            scene_boundaries=boundaries,
            beat_markers=markers,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class TimelineEditImpact:
    """Which downstream artifacts are invalidated by a timeline edit."""

    timeline_invalidated: bool = False
    scene_srt_invalidated: bool = False
    beat_json_invalidated: bool = False
    stage1_segments_invalidated: bool = False
    ltx_prompt_invalidated: bool = False
    render_plan_invalidated: bool = False


# ---------------------------------------------------------------------------
# Pure domain functions
# ---------------------------------------------------------------------------

_SEGMENT_TIME_EPSILON = 1e-9
_BEAT_TIME_EPSILON = 1e-6


def split_segment(
    segment: EditableTimelineSegment, at: float
) -> tuple[EditableTimelineSegment, EditableTimelineSegment]:
    """Split *segment* into two parts at time *at*.

    Raises ``ValueError`` if *at* is not strictly inside (start, end).
    """
    if at <= segment.start + _SEGMENT_TIME_EPSILON:
        raise ValueError(
            f"split point {at} must be > start ({segment.start})"
        )
    if at >= segment.end - _SEGMENT_TIME_EPSILON:
        raise ValueError(
            f"split point {at} must be < end ({segment.end})"
        )
    if at < segment.start or at > segment.end:
        raise ValueError(
            f"split point {at} is outside [{segment.start}, {segment.end}]"
        )
    left = EditableTimelineSegment(
        start=segment.start,
        end=at,
        kind=segment.kind,
        text=segment.text,
        lyrics_line=segment.lyrics_line,
        notes=segment.notes,
        is_draft=segment.is_draft,
    )
    right = EditableTimelineSegment(
        start=at,
        end=segment.end,
        kind=segment.kind,
        text=segment.text,
        lyrics_line=segment.lyrics_line,
        notes=segment.notes,
        is_draft=segment.is_draft,
    )
    return left, right


def merge_segments(
    segments: list[EditableTimelineSegment],
) -> EditableTimelineSegment:
    """Merge a list of *adjacent* segments into one.

    Segments must be non-overlapping and contiguous (end of each == start of next).
    Raises ``ValueError`` if fewer than 2 segments are provided or adjacency is
    broken.
    """
    if len(segments) < 2:
        raise ValueError(
            f"merge requires at least 2 segments, got {len(segments)}"
        )
    for i in range(len(segments) - 1):
        gap = segments[i + 1].start - segments[i].end
        if abs(gap) < _SEGMENT_TIME_EPSILON:
            continue  # adjacent
        if gap < 0:
            raise ValueError(
                f"segments overlap: [{segments[i].start}, {segments[i].end}] "
                f"and [{segments[i + 1].start}, {segments[i + 1].end}]"
            )
        raise ValueError(
            f"gap between segments: [{segments[i].start}, {segments[i].end}] "
            f"and [{segments[i + 1].start}, {segments[i + 1].end}]"
        )
    texts = [s.text for s in segments if s.text]
    merged_text = " ".join(texts)
    return EditableTimelineSegment(
        start=segments[0].start,
        end=segments[-1].end,
        kind=segments[0].kind,
        text=merged_text,
        lyrics_line=segments[0].lyrics_line,
        notes=segments[0].notes,
        is_draft=segments[0].is_draft,
    )


def validate_scene_boundaries(
    boundaries: list[SceneBoundary],
) -> list[SceneBoundary]:
    """Return *boundaries* sorted by start time, rejecting overlaps."""
    if not boundaries:
        return []
    sorted_b = sorted(boundaries, key=lambda b: b.start)
    for i in range(len(sorted_b) - 1):
        if sorted_b[i].end > sorted_b[i + 1].start + _SEGMENT_TIME_EPSILON:
            raise ValueError(
                f"scene boundaries overlap: [{sorted_b[i].start}, {sorted_b[i].end}] "
                f"and [{sorted_b[i + 1].start}, {sorted_b[i + 1].end}]"
            )
    return list(sorted_b)


def validate_beat_markers(
    markers: list[BeatMarker],
) -> list[BeatMarker]:
    """Return *markers* sorted by time, rejecting duplicates."""
    if not markers:
        return []
    sorted_m = sorted(markers, key=lambda m: m.time_s)
    for i in range(len(sorted_m) - 1):
        if abs(sorted_m[i + 1].time_s - sorted_m[i].time_s) < _BEAT_TIME_EPSILON:
            raise ValueError(
                f"duplicate beat time {sorted_m[i].time_s}"
            )
    return list(sorted_m)


def compute_edit_impact(
    before: TimelineSnapshot,
    after: TimelineSnapshot,
) -> TimelineEditImpact:
    """Compare two snapshots and return which artifacts are invalidated."""
    flags = {
        "timeline_invalidated": False,
        "scene_srt_invalidated": False,
        "beat_json_invalidated": False,
        "stage1_segments_invalidated": False,
        "ltx_prompt_invalidated": False,
        "render_plan_invalidated": False,
    }

    # Compare segments
    segs_before = list(before.segments)
    segs_after = list(after.segments)
    segments_changed = _segments_changed(segs_before, segs_after)
    segment_count_changed = len(segs_before) != len(segs_after)
    segment_times_changed = _segment_times_changed(segs_before, segs_after)
    segment_text_changed = _segment_texts_changed(segs_before, segs_after)
    segment_kind_changed = _segment_kinds_changed(segs_before, segs_after)
    segment_lyrics_changed = _segment_lyrics_changed(segs_before, segs_after)

    if segments_changed:
        flags["timeline_invalidated"] = True

    if segment_count_changed or segment_kind_changed or segment_times_changed:
        flags["stage1_segments_invalidated"] = True

    # Compare scene boundaries
    boundaries_changed = (sorted(before.scene_boundaries, key=lambda b: b.start)
                          != sorted(after.scene_boundaries, key=lambda b: b.start))
    if boundaries_changed:
        flags["scene_srt_invalidated"] = True
        flags["render_plan_invalidated"] = True

    # Compare beat markers
    markers_changed = (sorted(before.beat_markers, key=lambda m: m.time_s)
                       != sorted(after.beat_markers, key=lambda m: m.time_s))
    if markers_changed:
        flags["beat_json_invalidated"] = True

    # LTX prompt invalidated by text or lyrics changes
    if segment_text_changed or segment_lyrics_changed:
        flags["ltx_prompt_invalidated"] = True

    # Render plan invalidated by segment text changes
    if segment_text_changed:
        flags["render_plan_invalidated"] = True

    # Metadata changes invalidate render plan
    if before.metadata != after.metadata:
        flags["render_plan_invalidated"] = True

    return TimelineEditImpact(**flags)


# ---------------------------------------------------------------------------
# _private comparison helpers
# ---------------------------------------------------------------------------


def _segments_changed(a: list[EditableTimelineSegment], b: list[EditableTimelineSegment]) -> bool:
    return a != b


def _segment_times_changed(a: list[EditableTimelineSegment], b: list[EditableTimelineSegment]) -> bool:
    if len(a) != len(b):
        return True
    return any(sa.start != sb.start or sa.end != sb.end for sa, sb in zip(a, b))


def _segment_texts_changed(a: list[EditableTimelineSegment], b: list[EditableTimelineSegment]) -> bool:
    if len(a) != len(b):
        return True
    return any(sa.text != sb.text for sa, sb in zip(a, b))


def _segment_kinds_changed(a: list[EditableTimelineSegment], b: list[EditableTimelineSegment]) -> bool:
    if len(a) != len(b):
        return True
    return any(sa.kind != sb.kind for sa, sb in zip(a, b))


def _segment_lyrics_changed(a: list[EditableTimelineSegment], b: list[EditableTimelineSegment]) -> bool:
    if len(a) != len(b):
        return True
    return any(sa.lyrics_line != sb.lyrics_line for sa, sb in zip(a, b))
