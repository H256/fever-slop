from __future__ import annotations

from typing import TypedDict


class ContinuityHandoffPayload(TypedDict, total=False):
    """Pipeline-neutral metadata for an optional scene-boundary handoff."""

    source_scene: int
    last_frame_path: str
    last_frame_sha256: str
    transition: str
    source_clip_path: str
    source_clip_sha256: str
    extractor: str
