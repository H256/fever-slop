from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SeedVR2Pass:
    input_size: tuple[int, int]
    output_size: tuple[int, int]
    scale: float


def _even(value: float) -> int:
    return max(2, int(round(value / 2.0) * 2))


def resolve_target_resolution(
    source_width: int,
    source_height: int,
    *,
    target_width: int | None = None,
    target_height: int | None = None,
    default_scale: float = 2.0,
) -> tuple[int, int]:
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source dimensions must be positive")
    if default_scale <= 0:
        raise ValueError("default_scale must be positive")
    aspect = source_width / source_height
    if target_width is not None and target_height is not None:
        if target_width <= 0 or target_height <= 0:
            raise ValueError("target dimensions must be positive")
        target_aspect = target_width / target_height
        if abs(target_aspect / aspect - 1.0) > 0.02:
            raise ValueError("target dimensions have an incompatible aspect ratio")
        return _even(target_width), _even(target_height)
    if target_width is not None:
        if target_width <= 0:
            raise ValueError("target_width must be positive")
        return _even(target_width), _even(target_width / aspect)
    if target_height is not None:
        if target_height <= 0:
            raise ValueError("target_height must be positive")
        return _even(target_height * aspect), _even(target_height)
    return _even(source_width * default_scale), _even(source_height * default_scale)


def plan_seedvr2_passes(
    source_width: int,
    source_height: int,
    *,
    target_width: int | None = None,
    target_height: int | None = None,
    default_scale: float = 2.0,
    max_pass_scale: float = 1.5,
    max_ai_passes: int = 3,
) -> tuple[SeedVR2Pass, ...]:
    if max_pass_scale <= 1.0:
        raise ValueError("max_pass_scale must be greater than 1")
    if max_ai_passes < 1:
        raise ValueError("max_ai_passes must be positive")

    target = resolve_target_resolution(
        source_width,
        source_height,
        target_width=target_width,
        target_height=target_height,
        default_scale=default_scale,
    )
    current = (int(source_width), int(source_height))
    if target[0] < current[0] or target[1] < current[1]:
        raise ValueError("target resolution must not be smaller than the source")
    if current == target:
        return ()

    required_passes = math.ceil(math.log(target[0] / current[0], max_pass_scale))
    if required_passes > max_ai_passes:
        raise ValueError(
            f"target requires {required_passes} SeedVR2 passes, exceeding max_ai_passes={max_ai_passes}"
        )

    result: list[SeedVR2Pass] = []
    for index in range(max_ai_passes):
        if current == target:
            break
        remaining_ratio = target[0] / current[0]
        scale = min(max_pass_scale, remaining_ratio)
        next_size = target if remaining_ratio <= max_pass_scale else (
            _even(current[0] * scale), _even(current[1] * scale)
        )
        result.append(SeedVR2Pass(current, next_size, next_size[0] / current[0]))
        current = next_size
    if current != target:
        raise ValueError("SeedVR2 pass planner did not reach target resolution")
    return tuple(result)
