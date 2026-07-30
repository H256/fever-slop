from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rebuild service functions
# ---------------------------------------------------------------------------
# Each function takes a project directory path and rebuilds one downstream
# artifact from upstream sources. They are called in dependency order by
# the rebuild-plan-timeline job handler.


def rebuild_beat_json(project_dir: str | Path) -> dict[str, Any]:
    """Rebuild beat markers JSON from audio analysis."""
    _logger.warning("Rebuild of %s not yet implemented", "beat_json")
    return {"status": "deferred", "message": "rebuild not yet implemented"}


def rebuild_scene_srt(project_dir: str | Path) -> dict[str, Any]:
    """Rebuild scene SRT from timeline segments and scene boundaries."""
    _logger.warning("Rebuild of %s not yet implemented", "scene_srt")
    return {"status": "deferred", "message": "rebuild not yet implemented"}


def rebuild_stage1_segments(project_dir: str | Path) -> dict[str, Any]:
    """Rebuild Stage 1 segments JSON from timeline."""
    _logger.warning("Rebuild of %s not yet implemented", "stage1_segments")
    return {"status": "deferred", "message": "rebuild not yet implemented"}


def rebuild_ltx_prompt(project_dir: str | Path) -> dict[str, Any]:
    """Rebuild LTX prompt relay JSON from timeline text/lyrics."""
    _logger.warning("Rebuild of %s not yet implemented", "ltx_prompt")
    return {"status": "deferred", "message": "rebuild not yet implemented"}


def rebuild_render_plan(project_dir: str | Path) -> dict[str, Any]:
    """Rebuild render plan JSON from all upstream artifacts."""
    _logger.warning("Rebuild of %s not yet implemented", "render_plan")
    return {"status": "deferred", "message": "rebuild not yet implemented"}
