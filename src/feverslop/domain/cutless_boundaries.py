from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CutlessDuplicatePolicy = Literal["reject", "warn", "accept"]


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


@dataclass(frozen=True)
class CutlessBoundaryDiagnostic:
    """Measured evidence for one adjacent continuation boundary."""

    predecessor_segment_id: str
    successor_segment_id: str
    predecessor_last_frame_sha256: str
    successor_first_frame_sha256: str
    similarity: float
    timing_delta_frames: int

    def __post_init__(self) -> None:
        if not self.predecessor_segment_id or not self.successor_segment_id:
            raise ValueError("diagnostic segment IDs are required")
        for name in ("predecessor_last_frame_sha256", "successor_first_frame_sha256"):
            digest = str(getattr(self, name)).lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"{name} must be a SHA-256 digest")
        if not 0.0 <= float(self.similarity) <= 1.0:
            raise ValueError("boundary similarity must be between 0 and 1")
        if isinstance(self.timing_delta_frames, bool) or self.timing_delta_frames < 0:
            raise ValueError("timing_delta_frames must be non-negative")

    @property
    def is_proven_duplicate(self) -> bool:
        return (
            self.predecessor_last_frame_sha256.lower()
            == self.successor_first_frame_sha256.lower()
            and self.timing_delta_frames == 0
        )


@dataclass(frozen=True)
class CutlessAssemblyPlan:
    """Immutable cutless assembly decision; source clips are never rewritten."""

    segment_ids: tuple[str, ...]
    trim_first_frame_segments: tuple[str, ...]
    diagnostics: tuple[CutlessBoundaryDiagnostic, ...]
    outcome: Literal["accept", "warn"]
    crossfade: bool = False


def build_cutless_assembly_plan(
    segment_ids: list[str],
    boundaries: list[CutlessBoundary],
    diagnostics: list[CutlessBoundaryDiagnostic],
    *,
    duplicate_policy: CutlessDuplicatePolicy = "reject",
    minimum_similarity: float = 0.995,
) -> CutlessAssemblyPlan:
    """Build a hard-cut plan and apply policy to unproven boundaries."""
    if duplicate_policy not in {"reject", "warn", "accept"}:
        raise ValueError("duplicate_policy must be reject, warn, or accept")
    if not 0.0 <= float(minimum_similarity) <= 1.0:
        raise ValueError("minimum_similarity must be between 0 and 1")
    validate_cutless_chain(boundaries, segment_ids)
    if len(diagnostics) != len(boundaries):
        raise ValueError("every cutless boundary needs one diagnostic")

    trim_segments: list[str] = []
    has_warning = False
    for boundary, diagnostic in zip(boundaries, diagnostics, strict=True):
        expected = (boundary.predecessor_segment_id, boundary.successor_segment_id)
        actual = (diagnostic.predecessor_segment_id, diagnostic.successor_segment_id)
        if actual != expected:
            raise ValueError("cutless diagnostics must follow boundary order")
        if diagnostic.is_proven_duplicate:
            trim_segments.append(diagnostic.successor_segment_id)
            continue
        if diagnostic.similarity < minimum_similarity or diagnostic.timing_delta_frames:
            if duplicate_policy == "reject":
                raise ValueError(
                    f"cutless boundary rejected: {diagnostic.predecessor_segment_id} -> "
                    f"{diagnostic.successor_segment_id}",
                )
            has_warning = duplicate_policy == "warn"

    return CutlessAssemblyPlan(
        segment_ids=tuple(segment_ids),
        trim_first_frame_segments=tuple(trim_segments),
        diagnostics=tuple(diagnostics),
        outcome="warn" if has_warning else "accept",
    )


def validate_cutless_chain(boundaries: list[CutlessBoundary], segment_ids: list[str]) -> None:
    expected = list(segment_ids)
    if not boundaries:
        return
    if len(boundaries) != max(0, len(expected) - 1):
        raise ValueError("every adjacent segment pair needs one cutless boundary")
    for index, boundary in enumerate(boundaries):
        if (boundary.predecessor_segment_id, boundary.successor_segment_id) != (expected[index], expected[index + 1]):
            raise ValueError("cutless boundaries must follow segment order")
