from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite

from feverslop.domain.duration_capability import DurationCapability


@dataclass(frozen=True)
class SemanticRenderSegment:
    segment_id: str
    index: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    starts_with_anchor: bool


def split_semantic_action(
    *,
    action_id: str,
    start_seconds: float,
    duration_seconds: float,
    max_duration_seconds: float,
    fps: int,
    min_duration_seconds: float = 0.001,
    capability: DurationCapability | None = None,
) -> tuple[SemanticRenderSegment, ...]:
    """Split one semantic action into deterministic, continuously chainable chunks."""
    identifier = str(action_id).strip()
    start = float(start_seconds)
    duration = float(duration_seconds)
    maximum = float(max_duration_seconds)
    minimum = float(min_duration_seconds)
    if not identifier or not isfinite(start) or start < 0:
        raise ValueError("action_id and non-negative finite start_seconds are required")
    if not isfinite(duration) or duration <= 0:
        raise ValueError("duration_seconds must be finite and greater than zero")
    if not isfinite(maximum) or maximum <= 0:
        raise ValueError("max_duration_seconds must be finite and greater than zero")
    if not isfinite(minimum) or minimum <= 0 or minimum > maximum:
        raise ValueError("min_duration_seconds must be finite, positive, and <= max_duration_seconds")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("fps must be a positive integer")

    if capability is not None:
        fps = capability.fps
        maximum = capability.max_seconds
        minimum = max(minimum, capability.min_seconds)

    total_frames = round(duration * fps)
    maximum_frames = (
        capability.frames_for(maximum)
        if capability is not None
        else max(1, int(maximum * fps))
    )
    segment_count = max(1, ceil(total_frames / maximum_frames))
    if segment_count > 1 and total_frames / segment_count < minimum * fps:
        raise ValueError("duration would produce an unusably short continuation segment")

    if capability is None:
        base_frames, remainder = divmod(total_frames, segment_count)
        lengths = [base_frames + (1 if index < remainder else 0) for index in range(segment_count)]
    else:
        alignment = capability.frame_alignment
        offset = capability.frame_offset
        min_frames = round(minimum * fps)
        max_frames = maximum_frames
        candidates = []
        for first in range(offset, max_frames + 1, alignment):
            remaining = total_frames - first
            if remaining < max(0, segment_count - 1) * min_frames:
                break
            if remaining > max(0, segment_count - 1) * max_frames:
                continue
            if remaining % alignment == 0:
                candidates.append(first)
        if not candidates:
            raise ValueError("duration cannot be partitioned into aligned continuation segments")
        first = min(candidates, key=lambda value: abs(value - total_frames / segment_count))
        remaining = total_frames - first
        base_frames, remainder = divmod(remaining, segment_count - 1) if segment_count > 1 else (0, 0)
        lengths = [first]
        if segment_count > 1:
            lengths.extend(base_frames + (1 if index < remainder else 0) for index in range(segment_count - 1))
    segments: list[SemanticRenderSegment] = []
    elapsed_frames = 0
    for index, frames in enumerate(lengths, start=1):
        chunk_duration = frames / fps
        chunk_start = round(start + elapsed_frames / fps, 6)
        end = round(start + (elapsed_frames + frames) / fps, 6)
        segments.append(
            SemanticRenderSegment(
                segment_id=f"{identifier}-{index:04d}",
                index=index,
                start_seconds=chunk_start,
                end_seconds=end,
                duration_seconds=round(chunk_duration, 6),
                starts_with_anchor=index > 1,
            )
        )
        elapsed_frames += frames
    return tuple(segments)
