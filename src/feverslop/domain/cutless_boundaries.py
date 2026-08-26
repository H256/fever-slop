from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CutlessBoundary:
    predecessor_segment_id: str
    successor_segment_id: str
    boundary_frame_sha256: str

    def __post_init__(self) -> None:
        if not self.predecessor_segment_id or not self.successor_segment_id:
            raise ValueError("boundary segment IDs are required")
        digest = str(self.boundary_frame_sha256).lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("boundary_frame_sha256 must be a SHA-256 digest")


def validate_cutless_chain(boundaries: list[CutlessBoundary], segment_ids: list[str]) -> None:
    expected = list(segment_ids)
    if not boundaries:
        return
    if len(boundaries) != max(0, len(expected) - 1):
        raise ValueError("every adjacent segment pair needs one cutless boundary")
    for index, boundary in enumerate(boundaries):
        if (boundary.predecessor_segment_id, boundary.successor_segment_id) != (expected[index], expected[index + 1]):
            raise ValueError("cutless boundaries must follow segment order")
