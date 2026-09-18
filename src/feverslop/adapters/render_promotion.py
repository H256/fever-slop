"""Idempotent promotion of a terminal render result into scene state.

A render reaching a terminal (successful) state is recorded in the scene's
``SceneWorkflowManifest`` as a small promotion record keyed by a fingerprint of
the scene's render inputs. Re-promoting the same terminal result is a no-op;
a result bound to a different run/scene (a stale run) is refused with a clear
message instead of being silently applied.

The promotion unit is testable in isolation: it only touches the manifest
document and never requires a live ComfyUI server.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from feverslop.domain.prepared_workflow import SceneWorkflowManifest


class StaleRunError(ValueError):
    """A terminal result is bound to a different run/scene than the current one."""


# Decision outcomes for a promotion.
PROMOTE = "promote"
NOOP = "noop"
STALE = "stale"


def scene_fingerprint(manifest: SceneWorkflowManifest) -> str:
    """Deterministic fingerprint of the scene's render inputs.

    Two renders of the same scene with identical inputs produce the same
    fingerprint; changing the seed, resolution, frame budget, or workflow
    profile produces a different one.
    """
    payload = {
        "pipeline": manifest.pipeline,
        "seed": manifest.seed,
        "fps": manifest.fps,
        "frame_count": manifest.frame_count,
        "render_frame_count": manifest.render_frame_count,
        "workflow_profile": (
            manifest.consistency.workflow_profile if manifest.consistency else None
        ),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def decide_promotion(
    stored: dict[str, Any] | None, current_fingerprint: str
) -> str:
    """Pure decision: promote, no-op, or refuse (stale)."""
    if stored is None:
        return PROMOTE
    if stored.get("fingerprint") == current_fingerprint:
        return NOOP
    return STALE


class RenderPromotion:
    """Idempotent terminal-result promotion bound to a scene manifest."""

    def __init__(self, project_dir: str | Path) -> None:
        self.project_dir = Path(project_dir)

    def apply(
        self, manifest_path: str | Path, *, current_fingerprint: str
    ) -> str:
        """Promote the terminal result for ``manifest_path``.

        Returns the decision (``PROMOTE`` or ``NOOP``). Raises
        :class:`StaleRunError` when the stored record is bound to a different
        run/scene than ``current_fingerprint``.
        """
        path = Path(manifest_path)
        manifest = SceneWorkflowManifest.read(path)
        decision = decide_promotion(
            manifest.render_promotion, current_fingerprint
        )
        if decision == STALE:
            stored = manifest.render_promotion or {}
            raise StaleRunError(
                "Refusing to apply stale render result: stored fingerprint "
                f"{stored.get('fingerprint')} does not match current "
                f"{current_fingerprint} for scene {manifest.scene}"
            )
        if decision == NOOP:
            return NOOP
        record = {
            "fingerprint": current_fingerprint,
            "scene": manifest.scene,
            "pipeline": manifest.pipeline,
            "status": "terminal",
        }
        updated = replace(manifest, render_promotion=record)
        updated.write(path)
        return PROMOTE
