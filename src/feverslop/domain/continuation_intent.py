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


def continuation_intents_from_plan(actions: list[dict]) -> tuple[ContinuationIntent, ...]:
    """Normalize DSPy planner action records without inventing creative content."""
    result = []
    for action in actions:
        if not isinstance(action, dict):
            raise ValueError("planner actions must be objects")
        result.append(ContinuationIntent(
            action_id=str(action.get("action_id") or action.get("id") or "").strip(),
            requires_continuation=bool(action.get("requires_continuation", False)),
            rationale=str(action.get("rationale") or "").strip(),
            desired_duration_seconds=action.get("desired_duration_seconds"),
        ))
    return tuple(result)
