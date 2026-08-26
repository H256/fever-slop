from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, TypedDict


class ContinuityHandoffPayload(TypedDict, total=False):
    """Pipeline-neutral metadata for an optional scene-boundary handoff."""

    source_scene: int
    last_frame_path: str
    last_frame_sha256: str
    transition: str
    source_clip_path: str
    source_clip_sha256: str
    extractor: str


@dataclass(frozen=True)
class BoundaryFrameManifest:
    """Verified, project-relative artifact describing a continuation boundary."""

    source_clip_path: str
    source_clip_sha256: str
    frame_index: int
    extractor_revision: str
    frame_path: str
    frame_sha256: str

    @classmethod
    def create(
        cls,
        *,
        source_clip_path: str,
        source_clip_sha256: str,
        frame_index: int,
        extractor_revision: str,
        frame_path: str,
        frame_sha256: str,
    ) -> "BoundaryFrameManifest":
        def path(value: str, field: str) -> str:
            raw = str(value).strip().replace("\\", "/")
            parsed = PurePosixPath(raw)
            if not raw or PureWindowsPath(raw).drive or parsed.is_absolute() or ".." in parsed.parts:
                raise ValueError(f"{field} must be project-relative")
            return parsed.as_posix()

        def digest(value: str, field: str) -> str:
            raw = str(value).strip().lower()
            if len(raw) != 64 or any(char not in "0123456789abcdef" for char in raw):
                raise ValueError(f"{field} must be a SHA-256 hex digest")
            return raw

        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        revision = str(extractor_revision).strip()
        if not revision:
            raise ValueError("extractor_revision is required")
        return cls(
            source_clip_path=path(source_clip_path, "source_clip_path"),
            source_clip_sha256=digest(source_clip_sha256, "source_clip_sha256"),
            frame_index=frame_index,
            extractor_revision=revision,
            frame_path=path(frame_path, "frame_path"),
            frame_sha256=digest(frame_sha256, "frame_sha256"),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BoundaryFrameManifest":
        if not isinstance(payload, dict):
            raise ValueError("boundary frame manifest must be an object")
        return cls.create(**payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_clip_path": self.source_clip_path,
            "source_clip_sha256": self.source_clip_sha256,
            "frame_index": self.frame_index,
            "extractor_revision": self.extractor_revision,
            "frame_path": self.frame_path,
            "frame_sha256": self.frame_sha256,
        }

    def matches(self, *, source_clip_sha256: str, frame_sha256: str) -> bool:
        return (
            str(source_clip_sha256).strip().lower() == self.source_clip_sha256
            and str(frame_sha256).strip().lower() == self.frame_sha256
        )
