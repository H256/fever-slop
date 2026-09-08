from __future__ import annotations

from typing import Any, Literal, Protocol

RECOVERY_POLICY_VERSION = 1


def require_ready_scenes(scenes: list[dict[str, Any]]) -> None:
    blocked = [str(scene.get("scene") or scene.get("segment_id") or "unknown")
               for scene in scenes if (scene.get("readiness") or {}).get("status") == "blocked"]
    if blocked:
        raise ValueError("Scene preparation is incomplete; blocked scenes: " + ", ".join(blocked)
                         + ". Correct their inputs or explicitly replan them before rendering or finalizing.")
RecoveryStage = Literal['generate', 'repair', 'fallback']


class SceneRecoverySession(Protocol):
    input_fingerprint: str
    attempt_revision: int
    saved_result: dict[str, Any] | None

    @property
    def readiness(self) -> dict[str, Any]: ...

    @property
    def state(self) -> dict[str, Any]: ...

    def reserve(self, stage: RecoveryStage) -> bool: ...

    def finish(self, status: Literal['ready', 'blocked'], reason_codes: list[str], *, stage: str | None = None) -> dict[str, Any]: ...
