"""Ports for timeline document I/O.

Read/write ports isolate the application layer from storage adapters.
All reads return snapshot-safe copies (deep-copied or freshly allocated).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from feverslop.domain.timeline_editing import TimelineEditImpact

# ---------------------------------------------------------------------------
# Port protocols
# ---------------------------------------------------------------------------


class TimelineReadPort(Protocol):
    """Read-only access to timeline-editing artefacts."""

    def read_timeline(self) -> list[dict[str, Any]]:
        """Return a deep-copied timeline JSON payload."""
        ...

    def read_scene_srt(self) -> str | None:
        """Return current scene SRT text, or *None* if absent."""
        ...

    def read_beat_json(self) -> list[dict[str, Any]] | None:
        """Return current beat markers JSON, or *None* if absent."""
        ...

    def read_stage1_segments(self) -> list[dict[str, Any]] | None:
        """Return current Stage 1 segments JSON, or *None* if absent."""
        ...

    def read_ltx_prompt_relay(self) -> list[dict[str, Any]] | None:
        """Return current LTX prompt relay JSON, or *None* if absent."""
        ...

    def read_render_plan(self) -> dict[str, Any] | None:
        """Return current render plan JSON, or *None* if absent."""
        ...


class TimelineWritePort(Protocol):
    """Write-only access to timeline documents.

    Other artefacts (beat JSON, stage 1 segments, etc.) are only produced
    by the full pipeline — this port exposes direct writes for the core
    two documents that an editor modifies.
    """

    def write_timeline(self, data: list[dict[str, Any]]) -> None:
        """Persist a timeline JSON payload."""
        ...

    def write_scene_srt(self, content: str) -> None:
        """Persist scene SRT text."""
        ...


# ---------------------------------------------------------------------------
# AffectedArtifacts
# ---------------------------------------------------------------------------


@dataclass(frozen=False)
class AffectedArtifacts:
    """Which downstream artifacts may need rebuilding after an edit.

    Each flag corresponds to one pipeline stage whose output is stale.
    """

    timeline: bool = False
    scene_srt: bool = False
    beat_json: bool = False
    stage1_segments: bool = False
    ltx_prompt: bool = False
    render_plan: bool = False

    @staticmethod
    def from_timeline_edit_impact(impact: TimelineEditImpact) -> AffectedArtifacts:
        """Convert a domain ``TimelineEditImpact`` into an ``AffectedArtifacts``."""
        return AffectedArtifacts(
            timeline=impact.timeline_invalidated,
            scene_srt=impact.scene_srt_invalidated,
            beat_json=impact.beat_json_invalidated,
            stage1_segments=impact.stage1_segments_invalidated,
            ltx_prompt=impact.ltx_prompt_invalidated,
            render_plan=impact.render_plan_invalidated,
        )

    def any(self) -> bool:
        """Return ``True`` if any artifact is affected."""
        return any((
            self.timeline,
            self.scene_srt,
            self.beat_json,
            self.stage1_segments,
            self.ltx_prompt,
            self.render_plan,
        ))
