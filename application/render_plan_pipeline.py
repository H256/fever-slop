from __future__ import annotations

from typing import Any


class RenderPlanPipeline:
    """Application service boundary for final render plan assembly."""

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        return context
