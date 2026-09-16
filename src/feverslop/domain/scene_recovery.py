from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol

from feverslop.domain.h3_guidance import h3_block_guide

RECOVERY_POLICY_VERSION = 1


def require_ready_scenes(scenes: list[dict[str, Any]], *, project_path: Path | None = None) -> None:
    blocked: list[str] = []
    blocked_codes: list[list[str]] = []
    for scene in scenes:
        if (scene.get("readiness") or {}).get("status") == "blocked":
            scene_id = str(scene.get("scene") or scene.get("segment_id") or "unknown")
            blocked.append(scene_id)
            blocked_codes.append(scene.get("readiness", {}).get("reason_codes") or [])
    if blocked:
        # Build decoded guidance for each blocked scene.
        parts: list[str] = []
        for scene_id, codes in zip(blocked, blocked_codes):
            guide = h3_block_guide(codes)
            parts.append(f"{scene_id}: {guide}")
        suffix = ". Correct their inputs or explicitly replan them before rendering or finalizing."
        if project_path is not None:
            suffix += f"\nRun: uv run python main.py run {project_path} --replan-failed"
        raise ValueError("Scene preparation is incomplete; blocked scenes: "
                         + ", ".join(blocked) + suffix)
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
