from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REFERENCE_TYPE_ACTOR = "actor"
REFERENCE_TYPE_LOCATION = "location"
VALID_REFERENCE_TYPES = frozenset({REFERENCE_TYPE_ACTOR, REFERENCE_TYPE_LOCATION})


@dataclass(frozen=True)
class ReferenceImage:
    id: str
    type: str
    path: Path

    def __post_init__(self):
        if self.type not in VALID_REFERENCE_TYPES:
            raise ValueError(
                "ReferenceImage.type must be one of: "
                + ", ".join(sorted(VALID_REFERENCE_TYPES))
                + f" (got {self.type!r})",
            )
