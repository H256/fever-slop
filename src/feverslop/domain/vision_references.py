from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReferenceImage:
    id: str
    type: str
    path: Path
