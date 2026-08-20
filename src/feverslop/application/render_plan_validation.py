from __future__ import annotations

from collections.abc import Sequence
import math


def require_non_empty_render_plan(
    plan: Sequence[object],
    *,
    render_plan_path: object,
) -> None:
    """Raise a clear error when the parsed render plan contains no scenes."""
    if not plan:
        raise ValueError(f"Render plan is empty: {render_plan_path}")


def validate_render_plan_timeline(
    plan: Sequence[object],
    *,
    fps: int,
    render_plan_path: object,
) -> None:
    """Reject render plans whose scene frames overlap by more than one frame.

    Shared by the timeline exporters. Uses the same frame math and one-frame
    boundary tolerance as the MLT exporter's in-loop check, so plans the loop
    rejects as overlapping are rejected here before any project file is
    written. Entries without ``abs_start_seconds`` are anchored to the running
    cursor like the exporters' sequential entries.
    """
    fps = int(fps)
    cursor = 0
    intervals = []
    for index, entry in enumerate(plan, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Render plan entry {index} must be an object")
        scene_number = int(entry.get("scene") or entry.get("scene_number") or index)
        duration = float(entry.get("duration_seconds", 0.0))
        if duration <= 0:
            raise ValueError(f"Render plan scene {scene_number} has no positive duration")
        start_seconds = entry.get("abs_start_seconds")
        if start_seconds is not None:
            start_frame = max(0, round(float(start_seconds) * fps))
            end_frame = max(start_frame + 1, round((float(start_seconds) + duration) * fps))
        else:
            start_frame = cursor
            end_frame = start_frame + max(1, math.ceil(duration * fps))
        intervals.append((start_frame, end_frame, scene_number, index))
        cursor = max(cursor, end_frame)

    cursor = 0
    for start_frame, end_frame, scene_number, _index in sorted(
        intervals, key=lambda interval: (interval[0], interval[3])
    ):
        frames = end_frame - start_frame
        if start_frame < cursor:
            overlap = cursor - start_frame
            if overlap > 1:
                raise ValueError(
                    "Timeline export cannot represent overlapping render-plan entries: "
                    f"scene {scene_number} starts at frame {start_frame}, "
                    f"before frame {cursor}. Render plan: {render_plan_path}"
                )
            # Render-plan seconds are floating point values while the timeline
            # is frame-based. Treat a one-frame boundary discrepancy as a
            # contiguous cut instead of a real overlap.
            start_frame = cursor
        cursor = start_frame + frames
