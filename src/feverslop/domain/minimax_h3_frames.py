from __future__ import annotations

from feverslop.errors import FeverSlopValidationError

MINIMAX_H3_FPS = 24
MINIMAX_H3_MIN_DURATION_SECONDS = 4.0
MINIMAX_H3_MAX_DURATION_SECONDS = 15.0


def _next_valid_17n1(n: int) -> int:
    """Round *n* up to the next 17N+1 value.

    For n <= 0 returns 1.  For n >= 1 returns the smallest m >= n such that
    (m - 1) % 17 == 0.
    """
    if n <= 0:
        return 1
    remainder = (n - 1) % 17
    if remainder == 0:
        return n
    return n + (17 - remainder)


def _frames_from_duration(seconds: float) -> int:
    """Convert seconds to a 17N+1-constrained frame count at 24 fps."""
    raw_frames = max(5, round(seconds * MINIMAX_H3_FPS))
    return _next_valid_17n1(raw_frames)


def _duration_from_frames(frames: int) -> float:
    """Convert a 17N+1-constrained frame count to seconds at 24 fps."""
    return frames / 24.0


def _validate_duration(seconds: float) -> None:
    """Raise ``FeverSlopValidationError`` if *seconds* is outside the allowed range."""
    if seconds < MINIMAX_H3_MIN_DURATION_SECONDS or seconds > MINIMAX_H3_MAX_DURATION_SECONDS:
        raise FeverSlopValidationError(
            f"MiniMax H3 duration {seconds}s is outside allowed range "
            f"[{MINIMAX_H3_MIN_DURATION_SECONDS}s, {MINIMAX_H3_MAX_DURATION_SECONDS}s]"
        )


def _validate_frames(frames: int) -> None:
    """Raise ``FeverSlopValidationError`` if *frames* violates the 17N+1 constraint."""
    if frames < 1 or (frames - 1) % 17 != 0:
        raise FeverSlopValidationError(
            f"Frame count {frames} does not satisfy 17N+1 constraint (minimum 1)"
        )
