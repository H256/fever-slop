from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class GenerateRenderPlanContext:
    request: Any = None
    config: Any = None
    paths: Any = None
    app_config: Any = None
    video_settings: Any = None
    song_id: str = ""
    artifact_store: Any = None
    console: Any = None
    log_step: Any = None
    log_file: Any = None
    run_spinner: Any = None

    timeline_json: Path | None = None
    beat_json: Path | None = None
    scene_srt_raw: Path | None = None
    scene_srt: Path | None = None
    stage1_segments_json: Path | None = None
    ltx_prompt_relay_json: Path | None = None
    resolved_context_json: Path | None = None
    concept_prompts_json: Path | None = None
    scene_details_json: Path | None = None
    scene_prompts_json: Path | None = None
    render_plan_json: Path | None = None

    stem_files: dict[str, Path] | None = None
    timeline: Any = None
    beat_data: dict[str, Any] | None = None
    repaired_scenes: list[Any] | None = None
    stage1_segments: list[dict] | None = None
    global_context: dict[str, Any] | None = None
    concept_prompts: dict[str, Any] | None = None
    scene_details: dict[str, Any] | None = None
    render_plan: list[dict] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.extra[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.extra[key] = value

    def __iter__(self) -> Iterator[str]:
        yield from self.keys()

    def keys(self) -> set[str]:
        return {key for key in self.__dataclass_fields__ if key != "extra"} | set(self.extra)

    def update(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            self[key] = value

    def setdefault(self, key: str, default: Any = None) -> Any:
        try:
            value = self[key]
        except KeyError:
            self[key] = default
            return default
        if value is None:
            self[key] = default
            return default
        return value
