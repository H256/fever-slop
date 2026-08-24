from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FullAutoRequest:
    idea: str
    style: str
    music_style: str | None = None
    project_name: str | None = None
    projects_dir: Path = Path("projects")
    duration_seconds: float = 120.0
    width: int = 1280
    height: int = 704
    fps: int = 24
    language: str = "en"
    bpm: int | None = None
    keyscale: str | None = None
    seed: int = 0
    silent_mode: bool = False
    run_video_pipeline: bool = False
    runner_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SongSpec:
    title: str
    tags: str
    lyrics: str
    bpm: int
    duration_seconds: float
    language: str
    keyscale: str
    visual_story_idea: str
    visual_style: str
    music_style: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GeneratedSong:
    audio_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class ProjectScaffoldResult:
    project_dir: Path
    project_config_path: Path
    audio_path: Path
    lyrics_path: Path
    song_spec_path: Path
