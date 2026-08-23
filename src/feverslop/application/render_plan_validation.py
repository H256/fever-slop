from __future__ import annotations

from collections.abc import Sequence
import math

from feverslop.domain.subject_directives import validate_subject_directive_plan


def require_non_empty_render_plan(
    plan: Sequence[object],
    *,
    render_plan_path: object,
) -> None:
    """Raise a clear error when the parsed render plan contains no scenes."""
    if not plan:
        raise ValueError(f"Render plan is empty: {render_plan_path}")


def validate_render_plan_subject_directives(
    plan: Sequence[object],
    *,
    known_subject_ids: Sequence[str] = (),
    known_prop_ids: Sequence[str] = (),
    render_plan_path: object,
) -> None:
    """Validate opt-in subject directives while keeping legacy scenes readable."""
    for index, entry in enumerate(plan, start=1):
        if not isinstance(entry, dict) or entry.get("subject_directives") is None:
            continue
        from feverslop.domain.subject_directives import SubjectDirectivePlan

        try:
            directive_plan = SubjectDirectivePlan.from_dict(entry["subject_directives"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(
                f"Render plan scene {entry.get('scene', index)} has invalid subject directives "
                f"({render_plan_path}): {exc}"
            ) from exc
        issues = validate_subject_directive_plan(
            directive_plan,
            known_subject_ids=known_subject_ids,
            known_prop_ids=known_prop_ids,
        )
        if issues:
            raise ValueError(
                f"Render plan scene {entry.get('scene', index)} has invalid subject directives "
                f"({render_plan_path}): {'; '.join(issues)}"
            )


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
            if end_frame <= cursor:
                # Fully covered by earlier scenes: no contiguous placement left.
                raise ValueError(
                    "Timeline export cannot represent overlapping render-plan entries: "
                    f"scene {scene_number} ends at frame {end_frame}, "
                    f"before frame {cursor}. Render plan: {render_plan_path}"
                )
            # Scene-prompt seconds are floating point while the timeline is
            # frame-based: sub-frame boundary drift accumulates across scenes
            # until a scene starts before the cursor. Trim its tail so the cut
            # stays contiguous (one-frame overlaps are a special case).
            start_frame = cursor
            frames = end_frame - cursor
        cursor = start_frame + frames
