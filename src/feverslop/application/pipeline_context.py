from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from feverslop.adapters.reporting import NullReporter
from feverslop.domain.scene_duration_limits import ResolvedSceneDurationPolicy
from feverslop.ports.artifacts import ArtifactStore
from feverslop.ports.reporting import Reporter


@dataclass
class GenerateRenderPlanContext:
    request: Any = None
    config: Any = None
    paths: Any = None
    app_config: dict[str, Any] | None = None
    video_settings: Any = None
    song_id: str = ""
    scene_duration_policy: ResolvedSceneDurationPolicy | None = None
    artifact_store: ArtifactStore | None = None
    reporter: Reporter = NullReporter()
    console: Any = None
    log_step: Callable[[str], None] | None = None
    log_file: Callable[[str, Path], None] | None = None
    run_spinner: Callable[[str, Callable[[], Any]], Any] | None = None

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
    h3_prompts_json: Path | None = None
    render_plan_json: Path | None = None

    stem_files: dict[str, Path] | None = None
    timeline: list[dict] | None = None
    beat_data: dict[str, Any] | None = None
    repaired_scenes: list[Any] | None = None
    stage1_segments: list[dict] | None = None
    global_context: dict[str, Any] | None = None
    concept_prompts: dict[str, Any] | None = None
    scene_details: dict[str, Any] | None = None
    render_plan: list[dict] | None = None
    h3_prompts: list[dict] | None = None
    order: list[str] | None = None

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if not hasattr(self, key):
            raise KeyError(key)
        setattr(self, key, value)

    def __iter__(self) -> Iterator[str]:
        yield from self.keys()

    def keys(self) -> set[str]:
        return set(self.__dataclass_fields__)

    def update(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            self[key] = value

    def setdefault(self, key: str, default: Any = None) -> Any:
        value = self[key]
        if value is None:
            self[key] = default
            return default
        return value
