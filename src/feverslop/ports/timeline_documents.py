"""Ports for timeline document I/O.

Read/write ports isolate the application layer from storage adapters.
All reads return snapshot-safe copies (deep-copied or freshly allocated).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from feverslop.domain.timeline_editing import AffectedArtifacts

__all__ = ["AffectedArtifacts", "TimelineReadPort", "TimelineWritePort"]

# ---------------------------------------------------------------------------
# Port protocols
# ---------------------------------------------------------------------------


class TimelineReadPort(Protocol):
    """Read-only access to timeline-editing artefacts."""

    def read_timeline(self) -> list[Mapping[str, object]]:
        """Return a deep-copied timeline JSON payload."""
        ...

    def read_scene_srt(self) -> str | None:
        """Return current scene SRT text, or *None* if absent."""
        ...

    def read_beat_json(self) -> list[Mapping[str, object]] | None:
        """Return current beat markers JSON, or *None* if absent."""
        ...

    def read_stage1_segments(self) -> list[Mapping[str, object]] | None:
        """Return current Stage 1 segments JSON, or *None* if absent."""
        ...

    def read_ltx_prompt_relay(self) -> list[Mapping[str, object]] | None:
        """Return current LTX prompt relay JSON, or *None* if absent."""
        ...

    def read_render_plan(self) -> Mapping[str, object] | None:
        """Return current render plan JSON, or *None* if absent."""
        ...


class TimelineWritePort(Protocol):
    """Write-only access to timeline documents.

    Other artefacts (beat JSON, stage 1 segments, etc.) are only produced
    by the full pipeline — this port exposes direct writes for the core
    two documents that an editor modifies.
    """

    def write_timeline(self, data: list[Mapping[str, object]]) -> None:
        """Persist a timeline JSON payload."""
        ...

    def write_scene_srt(self, content: str) -> None:
        """Persist scene SRT text."""
        ...
