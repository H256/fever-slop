"""File-based adapter for timeline document ports.

Implements ``TimelineReadPort`` and ``TimelineWritePort`` against the
project's ``render/`` tree so that the application layer can edit
timeline.json and scene SRT while reading supporting artifacts
(beat JSON, stage-1 segments, LTX prompt relay, render plan).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def _read_json_file(path: Path) -> Any | None:
    """Read and parse a JSON file, returning ``None`` if missing."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return json.loads(text)


def _write_json_file(path: Path, data: Any) -> None:
    """Persist *data* as pretty-printed JSON, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_text_file(path: Path) -> str | None:
    """Read a text file, returning ``None`` if missing."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def _write_text_file(path: Path, content: str) -> None:
    """Persist *content* as text, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Artifact path constants
# ---------------------------------------------------------------------------

_TIMELINE_JSON = "render/timing/timeline.json"
_SCENE_SRT = "render/timing/scene_srt"
_BEAT_JSON = "render/timing/beat_json"
_STAGE1_SEGMENTS = "render/timing/stage1_segments.json"
_LTX_PROMPT_RELAY = "render/stage1/ltx_prompt_relay.json"
_RENDER_PLAN = "render/project_render_plan.json"


class ProjectTimelineDocuments:
    """Adapt project directory files to timeline document ports."""

    def __init__(self, project_dir: Path) -> None:
        self._project_dir = Path(project_dir).resolve()

    # -- TimelineReadPort --

    def read_timeline(self) -> list[dict[str, Any]]:
        raw = _read_json_file(self._project_dir / _TIMELINE_JSON)
        if raw is None:
            return []
        return copy.deepcopy(raw)

    def read_scene_srt(self) -> str | None:
        return _read_text_file(self._project_dir / _SCENE_SRT)

    def read_beat_json(self) -> list[dict[str, Any]] | None:
        raw = _read_json_file(self._project_dir / _BEAT_JSON)
        if raw is None:
            return None
        return copy.deepcopy(raw)

    def read_stage1_segments(self) -> list[dict[str, Any]] | None:
        raw = _read_json_file(self._project_dir / _STAGE1_SEGMENTS)
        if raw is None:
            return None
        return copy.deepcopy(raw)

    def read_ltx_prompt_relay(self) -> list[dict[str, Any]] | None:
        raw = _read_json_file(self._project_dir / _LTX_PROMPT_RELAY)
        if raw is None:
            return None
        return copy.deepcopy(raw)

    def read_render_plan(self) -> dict[str, Any] | None:
        raw = _read_json_file(self._project_dir / _RENDER_PLAN)
        if raw is None:
            return None
        return copy.deepcopy(raw)

    # -- TimelineWritePort --

    def write_timeline(self, data: list[dict[str, Any]]) -> None:
        _write_json_file(self._project_dir / _TIMELINE_JSON, data)

    def write_scene_srt(self, content: str) -> None:
        _write_text_file(self._project_dir / _SCENE_SRT, content)
