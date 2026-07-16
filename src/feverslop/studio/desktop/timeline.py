from __future__ import annotations


MINIMUM_TRIM_SECONDS = 0.04


def normalize_trim(start: float, end: float, *, duration: float) -> tuple[float, float]:
    maximum = max(0.0, float(duration))
    normalized_start = min(max(0.0, float(start)), maximum)
    normalized_end = min(max(normalized_start + MINIMUM_TRIM_SECONDS, float(end)), maximum)
    if maximum and normalized_end <= normalized_start:
        normalized_start = max(0.0, maximum - MINIMUM_TRIM_SECONDS)
        normalized_end = maximum
    return normalized_start, normalized_end


def format_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(float(seconds) * 1000))
    minutes, remainder = divmod(milliseconds, 60_000)
    whole_seconds, fraction = divmod(remainder, 1000)
    return f"{minutes:02d}:{whole_seconds:02d}.{fraction:03d}"
