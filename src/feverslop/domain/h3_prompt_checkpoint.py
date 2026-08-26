from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

H3_CHECKPOINT_SCHEMA = "feverslop.h3-prompt-checkpoint.v1"
H3CheckpointStatus = Literal["good", "bad_exhausted", "unjudged"]


@dataclass(frozen=True)
class H3PromptCheckpointInput:
    scene_number: int
    segment_id: str
    segment: Mapping[str, Any] = field(repr=False)
    concept: str = field(repr=False)
    scene_details: Mapping[str, Any] = field(repr=False)
    global_context: Mapping[str, Any] = field(repr=False)
    mode: str
    video_type: str
    audio_paths: Mapping[str, Path] = field(default_factory=dict, repr=False)
    generator_revision: Mapping[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class H3PromptCheckpoint:
    path: Path
    scene_number: int
    scene_id: str
    segment_id: str
    status: H3CheckpointStatus
    input_fingerprint: str
    generated: dict[str, Any] = field(repr=False)
    stage_fingerprints: Mapping[str, str] = field(default_factory=dict, repr=False)


def checkpoint_status(generated: Mapping[str, Any]) -> H3CheckpointStatus:
    judge = generated.get("prompt_judge")
    verdict = str(judge.get("verdict") or "").strip().lower() if isinstance(judge, Mapping) else ""
    if verdict == "good":
        return "good"
    if verdict == "bad":
        return "bad_exhausted"
    return "unjudged"
