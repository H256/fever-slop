from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RelayRange:
    """A relay segment over the half-open frame range ``[start, end_exclusive)``.

    ``end_exclusive`` follows the Python slice convention: the segment covers
    :meth:`length` frames, i.e. ``start .. end_exclusive - 1``. Keeping the
    convention in one value object replaces the ad-hoc ``frame_start``/
    ``frame_end`` clamping that previously lived in a comment (M-22 / #1182).

    The value is stored verbatim; :meth:`clamp` is the single place that
    normalizes raw (possibly negative or out-of-bounds) values to a valid
    ``[0, frame_count)`` range, so callers may construct from untrusted dict
    input without a second bound check.
    """

    start: int
    end_exclusive: int

    @property
    def length(self) -> int:
        """Number of frames covered: ``end_exclusive - start``."""
        return self.end_exclusive - self.start

    def clamp(self, frame_count: int) -> RelayRange | None:
        """Clamp to ``[0, frame_count)`` with at least one frame.

        Mirrors the previous ``_clamp_relay_segment`` producer semantics:
        ``start`` is clamped to ``[0, frame_count - 1]`` and ``end_exclusive``
        to ``[start + 1, frame_count]``. The result always holds at least one
        frame, so the ``None`` return is defensive only (a degenerate negative
        ``frame_count``); callers may still treat the return as possibly empty.
        """
        start = max(0, min(self.start, frame_count - 1))
        end = max(start + 1, min(self.end_exclusive, frame_count))
        if end <= start:
            return None
        return RelayRange(start, end)

    @classmethod
    def from_dict(cls, data: dict, *, start_key: str = "frame_start", end_key: str = "frame_end") -> RelayRange:
        """Build from a raw ``frame_start``/``frame_end`` dict at the boundary."""
        return cls(int(data[start_key]), int(data[end_key]))

    def to_dict(self, *, start_key: str = "frame_start", end_key: str = "frame_end") -> dict:
        """Emit the raw boundary dict for plan/relay payloads."""
        return {start_key: self.start, end_key: self.end_exclusive}
