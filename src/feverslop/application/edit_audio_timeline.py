"""Application use cases for audio timeline editing.

Thin orchestration layer: reads via ports, applies domain logic, writes via ports.

Public API
----------
The four symbols below constitute the application-layer use-case API surface.
They are wired into ``TimelineStudioService`` at runtime (see
``feverslop.studio.desktop.viewmodels.timeline``) and validated by the
dedicated test suite (``tests/test_edit_audio_timeline_app.py``).

Keep this file and its tests: the tests exercise the port protocols and domain
functions even though production code reaches these use cases indirectly
through the studio service.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any

from feverslop.domain.timeline_editing import (
    EditableTimelineSegment,
    TimelineEditImpact,
    TimelineSnapshot,
    compute_edit_impact,
)
from feverslop.ports.timeline_documents import (
    AffectedArtifacts,
    TimelineReadPort,
    TimelineWritePort,
)

__all__ = [
    "EditAudioTimeline",
    "SaveTimelines",
    "RebuildDownstreamArtifacts",
    "ComputeEditImpact",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class EditError(RuntimeError):
    """Raised when a timeline edit fails validation or I/O."""


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EditResult:
    """Outcome of a timeline edit operation."""

    after_snapshot: TimelineSnapshot
    impact: AffectedArtifacts
    timestamp: datetime.datetime


# ---------------------------------------------------------------------------
# ComputeEditImpact (application wrapper)
# ---------------------------------------------------------------------------


def ComputeEditImpact(
    before: TimelineSnapshot,
    after: TimelineSnapshot,
) -> AffectedArtifacts:
    """Compare two snapshots and return which downstream artefacts are stale.

    Delegates to domain ``compute_edit_impact`` and maps the result into the
    application-layer ``AffectedArtifacts`` type.
    """
    domain_impact: TimelineEditImpact = compute_edit_impact(before, after)
    return AffectedArtifacts.from_timeline_edit_impact(domain_impact)


# ---------------------------------------------------------------------------
# SaveTimelines
# ---------------------------------------------------------------------------


class SaveTimelines:
    """Persist current read-port state into the write port.

    Orchestrates: read timeline + scene_srt → write both.
    """

    def __init__(self, read_port: TimelineReadPort, write_port: TimelineWritePort):
        self.read_port = read_port
        self.write_port = write_port

    def execute(self) -> None:
        timeline_data: list[dict[str, Any]] = self.read_port.read_timeline()
        self.write_port.write_timeline(timeline_data)
        srt: str | None = self.read_port.read_scene_srt()
        if srt is not None:
            self.write_port.write_scene_srt(srt)


# ---------------------------------------------------------------------------
# RebuildDownstreamArtifacts
# ---------------------------------------------------------------------------


_ARTIFACT_JOB_KEYS: dict[str, str] = {
    "timeline": "timeline",
    "scene_srt": "scene_srt",
    "beat_json": "beat_json",
    "stage1_segments": "stage1_segments",
    "ltx_prompt": "ltx_prompt",
    "render_plan": "render_plan",
}


def RebuildDownstreamArtifacts(impact: AffectedArtifacts) -> list[str]:
    """Return list of pipeline job keys to queue for invalidated artifacts.

    Each ``True`` flag on *impact* maps to one job key string.
    """
    jobs: list[str] = []
    for field, key in _ARTIFACT_JOB_KEYS.items():
        if getattr(impact, field, False):
            jobs.append(key)
    return jobs


# ---------------------------------------------------------------------------
# EditAudioTimeline (main use case)
# ---------------------------------------------------------------------------


class EditAudioTimeline:
    """Orchestrate a timeline segment edit.

    Lifecycle::

        1. Read current state from read_port
        2. Apply *changes* to segment at *segment_index*
        3. Validate via domain models (frozen dataclasses do this in __post_init__)
        4. Write updated state via write_port
        5. Compute edit impact (before vs after snapshots)
        6. Return EditResult
    """

    def __init__(
        self,
        read_port: TimelineReadPort,
        write_port: TimelineWritePort,
    ):
        self.read_port = read_port
        self.write_port = write_port

    def edit(
        self,
        segment_index: int,
        changes: dict[str, Any],
        timestamp: datetime.datetime | None = None,
    ) -> EditResult:
        # 1. Read current state
        timeline_data = self.read_port.read_timeline()

        if not timeline_data:
            raise EditError("timeline is empty — nothing to edit")

        snapshot_data = timeline_data[0]
        before_snapshot = TimelineSnapshot.from_json(snapshot_data)

        # Bounds check
        count = len(before_snapshot.segments)
        if segment_index < 0 or segment_index >= count:
            raise EditError(
                f"segment_index {segment_index} out of range "
                f"[0, {count})"
            )

        # 2. Apply changes to the target segment
        segments = list(before_snapshot.segments)
        original = segments[segment_index]

        try:
            new_fields = dict(original.__dict__)
            new_fields.update(changes)
            segments[segment_index] = EditableTimelineSegment(**new_fields)
        except ValueError as exc:
            raise EditError(f"validation failed: {exc}") from exc

        # 3. Build after snapshot (domain validation happened in step 2)
        after_data = before_snapshot.to_json()
        after_data["segments"] = [
            {
                "start": s.start,
                "end": s.end,
                "kind": s.kind,
                "text": s.text,
                "lyrics_line": s.lyrics_line,
                "notes": s.notes,
                "is_draft": s.is_draft,
            }
            for s in segments
        ]
        after_snapshot = TimelineSnapshot.from_json(after_data)

        # 4. Write
        try:
            self.write_port.write_timeline([after_snapshot.to_json()])
        except OSError as exc:
            raise EditError(f"write failed: {exc}") from exc

        # 5. Compute impact
        impact = ComputeEditImpact(before_snapshot, after_snapshot)

        # 6. Return result
        ts = timestamp or datetime.datetime.now(datetime.timezone.utc)
        return EditResult(
            after_snapshot=after_snapshot,
            impact=impact,
            timestamp=ts,
        )
