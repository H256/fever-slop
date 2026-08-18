from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


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
