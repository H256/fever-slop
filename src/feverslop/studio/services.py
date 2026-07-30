from __future__ import annotations

from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Rebuild service functions
# ---------------------------------------------------------------------------
# Each function takes a project directory path and rebuilds one downstream
# artifact from upstream sources. They are called in dependency order by
# the rebuild-plan-timeline job handler.


def rebuild_beat_json(project_dir: str | Path) -> dict[str, Any]:
    """Rebuild beat markers JSON from audio analysis."""
    raise NotImplementedError(
        "rebuild_beat_json: real pipeline wiring TBD. "
        "Should analyze audio and regenerate beat markers."
    )


def rebuild_scene_srt(project_dir: str | Path) -> dict[str, Any]:
    """Rebuild scene SRT from timeline segments and scene boundaries."""
    raise NotImplementedError(
        "rebuild_scene_srt: real pipeline wiring TBD. "
        "Should regenerate scene SRT from current timeline data."
    )


def rebuild_stage1_segments(project_dir: str | Path) -> dict[str, Any]:
    """Rebuild Stage 1 segments JSON from timeline."""
    raise NotImplementedError(
        "rebuild_stage1_segments: real pipeline wiring TBD. "
        "Should regenerate Stage 1 segments from timeline data."
    )


def rebuild_ltx_prompt(project_dir: str | Path) -> dict[str, Any]:
    """Rebuild LTX prompt relay JSON from timeline text/lyrics."""
    raise NotImplementedError(
        "rebuild_ltx_prompt: real pipeline wiring TBD. "
        "Should regenerate LTX prompts from updated lyrics/text."
    )


def rebuild_render_plan(project_dir: str | Path) -> dict[str, Any]:
    """Rebuild render plan JSON from all upstream artifacts."""
    raise NotImplementedError(
        "rebuild_render_plan: real pipeline wiring TBD. "
        "Should regenerate render plan from beat, scene, stage1, and prompt data."
    )
