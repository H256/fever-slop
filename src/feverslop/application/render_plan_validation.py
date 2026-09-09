from __future__ import annotations

import math
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def require_non_empty_render_plan(
    plan: Sequence[object],
    *,
    render_plan_path: object,
) -> None:
    """Raise a clear error when the parsed render plan contains no scenes."""
    if not plan:
        raise ValueError(f"Render plan is empty: {render_plan_path}")


def load_render_plan_entries(
    render_plan_path: str | Path,
) -> list[tuple[dict[str, Any], int, float, float | None]]:
    """Load and validate the common render-plan entry contract."""
    path = Path(render_plan_path)
    plan = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(plan, dict):
        plan = plan.get("shots") or plan.get("scenes") or []
    if not isinstance(plan, list):
        raise ValueError(f"Render plan must be a JSON list: {render_plan_path}")
    require_non_empty_render_plan(plan, render_plan_path=render_plan_path)

    entries = []
    for index, entry in enumerate(plan, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Render plan entry {index} must be an object")
        scene_number = int(entry.get("scene") or entry.get("scene_number") or index)
        duration = float(entry.get("duration_seconds", 0.0))
        if duration <= 0:
            raise ValueError(f"Render plan scene {scene_number} has no positive duration")
        start_seconds = entry.get("abs_start_seconds")
        entries.append((entry, scene_number, duration, None if start_seconds is None else float(start_seconds)))
    return entries


def compute_timeline_intervals(
    entries: Sequence[tuple[dict[str, Any], int, float, float | None]],
    *,
    fps: int,
) -> list[tuple[dict[str, Any], int, float, float | None, int, int]]:
    """Return entries in the canonical MLT order with frame intervals."""
    indexed = list(enumerate(entries, start=1))
    indexed.sort(key=lambda item: (item[1][3] is None, item[1][3] or 0.0, item[0]))
    cursor = 0
    intervals = []
    for _index, (entry, scene_number, duration, start_seconds) in indexed:
        if start_seconds is not None:
            start_frame = max(0, round(start_seconds * int(fps)))
            end_frame = max(start_frame + 1, round((start_seconds + duration) * int(fps)))
        else:
            start_frame = cursor
            end_frame = start_frame + max(1, math.ceil(duration * int(fps)))
        intervals.append((_index, entry, scene_number, duration, start_seconds, start_frame, end_frame))
        cursor = max(cursor, end_frame)
    return intervals


def validate_render_plan_timeline(
    plan: Sequence[object],
    *,
    fps: int,
    render_plan_path: object,
) -> None:
    """Reject render plans whose scenes cannot form a contiguous timeline.

    Shared by the timeline exporters. Uses the same frame math and overlap
    rule as the exporters' in-loop checks, so plans the loops reject are
    rejected here before any project file is written. Scene prompts store
    start, duration, and end as floating point seconds that drift apart by
    sub-frame amounts, so accumulated boundary drift makes a scene start
    before the running cursor; such a scene is auto-corrected by trimming its
    tail at the cursor (a one-frame boundary overlap is the simplest case).
    Only a scene fully covered by earlier scenes, with its end at or before
    the cursor, is rejected. Entries without ``abs_start_seconds`` are
    anchored to the running cursor like the exporters' sequential entries.
    """
    entries = []
    for index, entry in enumerate(plan, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Render plan entry {index} must be an object")
        scene_number = int(entry.get("scene") or entry.get("scene_number") or index)
        duration = float(entry.get("duration_seconds", 0.0))
        start_seconds = entry.get("abs_start_seconds")
        entries.append((entry, scene_number, duration, None if start_seconds is None else float(start_seconds)))
    cursor = 0
    for _index, _entry, scene_number, _duration, _start_seconds, start_frame, end_frame in compute_timeline_intervals(entries, fps=fps):
        frames = end_frame - start_frame
        if start_frame < cursor:
            if end_frame <= cursor:
                # Fully covered by earlier scenes: no contiguous placement left.
                raise ValueError(
                    "Timeline export cannot represent overlapping render-plan entries: "
                    f"scene {scene_number} ends at frame {end_frame}, "
                    f"before frame {cursor}. Render plan: {render_plan_path}",
                )
            # Scene-prompt seconds are floating point while the timeline is
            # frame-based: sub-frame boundary drift accumulates across scenes
            # until a scene starts before the cursor. Trim its tail so the cut
            # stays contiguous (one-frame overlaps are a special case).
            start_frame = cursor
            frames = end_frame - cursor
        cursor = start_frame + frames
