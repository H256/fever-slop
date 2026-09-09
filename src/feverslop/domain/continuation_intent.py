from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContinuationIntent:
    """Planner output declaring whether an action must span technical shots."""

    action_id: str
    requires_continuation: bool
    rationale: str = ""
    desired_duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if not str(self.action_id).strip():
            raise ValueError("action_id is required")
        if self.desired_duration_seconds is not None and self.desired_duration_seconds <= 0:
            raise ValueError("desired_duration_seconds must be positive")
