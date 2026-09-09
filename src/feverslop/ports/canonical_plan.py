from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from feverslop.domain.canonical_plan_regeneration import CanonicalPlanSnapshot


class CanonicalPlanStore(Protocol):
    def capture_regeneration(self) -> CanonicalPlanSnapshot: ...

    def commit_regeneration(
        self,
        snapshot: CanonicalPlanSnapshot,
        scenes: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> Path: ...
