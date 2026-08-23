from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from feverslop.domain.visual_consistency import ReferenceAnchor

ReferenceKey = tuple[str, str]


@dataclass(frozen=True)
class ReferenceManifestSnapshot:
    actors: Mapping[ReferenceKey, ReferenceAnchor]
    locations: Mapping[ReferenceKey, ReferenceAnchor]
    revision: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "actors",
            MappingProxyType(_validated_anchors(self.actors, "actor")),
        )
        object.__setattr__(
            self,
            "locations",
            MappingProxyType(_validated_anchors(self.locations, "location")),
        )
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise ValueError("revision is required")


class ReferenceManifestPort(Protocol):
    def load(self, project_id: str) -> ReferenceManifestSnapshot:
        """Load immutable project reference anchors and their revision."""


class PreviousFramePort(Protocol):
    def extract_last_frame(
        self,
        video_path: Path,
        output_path: Path,
    ) -> Path:
        """Extract the final video frame to a PNG path."""


def _validated_anchors(
    anchors: Mapping[ReferenceKey, ReferenceAnchor],
    kind: str,
) -> dict[ReferenceKey, ReferenceAnchor]:
    if not isinstance(anchors, Mapping):
        raise TypeError(f"{kind} anchors must be a mapping")
    validated: dict[ReferenceKey, ReferenceAnchor] = {}
    for key, anchor in anchors.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or not all(isinstance(value, str) and value.strip() for value in key)
        ):
            raise ValueError(f"{kind} anchor keys must be (semantic_id, look_id)")
        if not isinstance(anchor, ReferenceAnchor) or anchor.kind != kind:
            raise ValueError(f"{kind} anchors must have kind {kind}")
        if key != (anchor.id, anchor.look_id):
            raise ValueError(f"{kind} anchor key must match its id and look id")
        validated[key] = anchor
    return validated
