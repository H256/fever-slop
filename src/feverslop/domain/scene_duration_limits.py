from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
import math
from pathlib import Path
from typing import Mapping, Sequence

from feverslop.domain.ltx_rendering import round_down_8n1
from feverslop.errors import FeverSlopValidationError


@dataclass(frozen=True)
class ResolvedSceneDurationPolicy:
    requested_min_seconds: float
    requested_max_seconds: float
    effective_min_seconds: float
    effective_max_seconds: float
    max_render_duration_seconds: float | None
    max_render_frames: int | None
    max_scene_frames: int | None
    fps: int
    preroll_frames: int
    tail_frames: int
    limiting_workflow: str | None

    @property
    def clamped(self) -> bool:
        return (
            self.effective_min_seconds != self.requested_min_seconds
            or self.effective_max_seconds != self.requested_max_seconds
        )


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise FeverSlopValidationError(f"{name} must be finite and greater than zero")
    return result


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, int):
        if value < 0:
            raise FeverSlopValidationError(f"{name} must be a non-negative integer")
        return value
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise FeverSlopValidationError(f"{name} must be a non-negative integer")
    return int(numeric)


def resolve_scene_duration_policy(
    *,
    requested_min_seconds: float,
    requested_max_seconds: float,
    fps: int,
    preroll_frames: int,
    tail_frames: int,
    round_render_frames_to_8n1: bool,
    workflow_limits: Mapping[str, float],
    workflow_paths: Sequence[str | Path],
    default_max_render_duration_seconds: float | None,
) -> ResolvedSceneDurationPolicy:
    requested_min = _positive_finite(requested_min_seconds, "requested_min_seconds")
    requested_max = _positive_finite(requested_max_seconds, "requested_max_seconds")
    if requested_min > requested_max:
        raise FeverSlopValidationError(
            "requested_min_seconds must be less than or equal to requested_max_seconds"
        )

    resolved_fps = _nonnegative_integer(fps, "fps")
    if resolved_fps == 0:
        raise FeverSlopValidationError("fps must be greater than zero")
    preroll = _nonnegative_integer(preroll_frames, "preroll_frames")
    tail = _nonnegative_integer(tail_frames, "tail_frames")

    normalized_limits: dict[str, float] = {}
    for workflow, duration in workflow_limits.items():
        basename = Path(str(workflow).strip()).name.casefold()
        if not basename:
            raise FeverSlopValidationError("workflow limit names must be non-empty")
        resolved_duration = _positive_finite(duration, "workflow max render duration")
        previous = normalized_limits.get(basename)
        normalized_limits[basename] = (
            resolved_duration if previous is None else min(previous, resolved_duration)
        )

    default_duration = (
        None
        if default_max_render_duration_seconds is None
        else _positive_finite(
            default_max_render_duration_seconds,
            "default_max_render_duration_seconds",
        )
    )

    workflow_basenames: list[str] = []
    for workflow_path in workflow_paths:
        basename = Path(str(workflow_path).strip()).name.casefold()
        if basename:
            workflow_basenames.append(basename)

    candidates: list[tuple[float, str | None]] = []
    for basename in workflow_basenames:
        duration = normalized_limits.get(basename, default_duration)
        if duration is not None:
            candidates.append((duration, basename))
    if not workflow_basenames and default_duration is not None:
        candidates.append((default_duration, None))

    if not candidates:
        return ResolvedSceneDurationPolicy(
            requested_min_seconds=requested_min,
            requested_max_seconds=requested_max,
            effective_min_seconds=requested_min,
            effective_max_seconds=requested_max,
            max_render_duration_seconds=None,
            max_render_frames=None,
            max_scene_frames=None,
            fps=resolved_fps,
            preroll_frames=preroll,
            tail_frames=tail,
            limiting_workflow=None,
        )

    max_render_duration, limiting_workflow = min(candidates, key=lambda item: item[0])
    render_intervals = (
        Decimal(str(max_render_duration)) * Decimal(resolved_fps)
    ).to_integral_value(rounding=ROUND_FLOOR)
    max_render_frames = int(render_intervals) + 1
    if round_render_frames_to_8n1:
        max_render_frames = round_down_8n1(max_render_frames)
    max_scene_frames = max_render_frames - preroll - tail
    if max_scene_frames < 1:
        raise FeverSlopValidationError(
            f"Render budget for {limiting_workflow or 'the default workflow'} cannot fit "
            f"{preroll} pre-roll frames, {tail} tail frames, and one scene frame"
        )

    capped_max_decimal = Decimal(max_scene_frames) / Decimal(resolved_fps)
    requested_max_decimal = Decimal(str(requested_max))
    effective_max = (
        requested_max
        if capped_max_decimal >= requested_max_decimal
        else float(capped_max_decimal)
    )
    effective_min = min(requested_min, effective_max)
    return ResolvedSceneDurationPolicy(
        requested_min_seconds=requested_min,
        requested_max_seconds=requested_max,
        effective_min_seconds=effective_min,
        effective_max_seconds=effective_max,
        max_render_duration_seconds=max_render_duration,
        max_render_frames=max_render_frames,
        max_scene_frames=max_scene_frames,
        fps=resolved_fps,
        preroll_frames=preroll,
        tail_frames=tail,
        limiting_workflow=limiting_workflow,
    )


def validate_render_frame_budget(
    *,
    scene_number: int,
    render_frame_count: int,
    fps: int,
    workflow_path: str | Path,
    max_render_frames: int | None,
    max_render_duration_seconds: float | None,
) -> None:
    if max_render_frames is None:
        return

    resolved_fps = _nonnegative_integer(fps, "fps")
    if resolved_fps == 0:
        raise FeverSlopValidationError("fps must be greater than zero")
    resolved_max_render_frames = _nonnegative_integer(
        max_render_frames,
        "max_render_frames",
    )
    if resolved_max_render_frames == 0:
        raise FeverSlopValidationError("max_render_frames must be greater than zero")
    if max_render_duration_seconds is not None:
        configured_duration = _positive_finite(
            max_render_duration_seconds,
            "max_render_duration_seconds",
        )
        allowed_interval = Decimal(resolved_max_render_frames - 1) / Decimal(resolved_fps)
        if allowed_interval > Decimal(str(configured_duration)):
            raise FeverSlopValidationError(
                "max_render_frames exceeds max_render_duration_seconds"
            )
    if int(render_frame_count) <= resolved_max_render_frames:
        return

    required_seconds = (int(render_frame_count) - 1) / float(resolved_fps)
    allowed_seconds = (resolved_max_render_frames - 1) / float(resolved_fps)
    workflow = Path(workflow_path).name
    raise FeverSlopValidationError(
        f"Scene {int(scene_number)} requires {int(render_frame_count)} render frames "
        f"({required_seconds:.3f}s), but {workflow} is limited to "
        f"{resolved_max_render_frames} frames ({allowed_seconds:.3f}s). "
        "Regenerate the render plan with the active workflow limit."
    )
