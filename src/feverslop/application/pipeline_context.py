from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from feverslop.ports.artifacts import ArtifactStore
from feverslop.ports.reporting import NullReporter, Reporter


@dataclass
class GenerateRenderPlanContext:
    request: object = None
    config: object = None
    paths: object = None
    app_config: object = None
    video_settings: object = None
    song_id: str = ""
    artifact_store: ArtifactStore | object = None
    reporter: Reporter = NullReporter()
    console: object = None
    log_step: Callable[[str], None] | None = None
    log_file: Callable[[str, Path], None] | None = None
    run_spinner: Callable[[str, Callable[[], object]], object] | None = None

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
    timeline: object = None
    beat_data: dict[str, object] | None = None
    repaired_scenes: list[object] | None = None
    stage1_segments: list[dict] | None = None
    global_context: dict[str, object] | None = None
    concept_prompts: dict[str, object] | None = None
    scene_details: dict[str, object] | None = None
    render_plan: list[dict] | None = None
    order: list[str] | None = None

    def __getitem__(self, key: str) -> object:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: object) -> None:
        if not hasattr(self, key):
            raise KeyError(key)
        setattr(self, key, value)

    def __iter__(self) -> Iterator[str]:
        yield from self.keys()

    def keys(self) -> set[str]:
        return set(self.__dataclass_fields__)

    def update(self, values: dict[str, object]) -> None:
        for key, value in values.items():
            self[key] = value

    def setdefault(self, key: str, default: object = None) -> object:
        value = self[key]
        if value is None:
            self[key] = default
            return default
        return value
