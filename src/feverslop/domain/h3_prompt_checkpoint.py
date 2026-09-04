from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

H3_CHECKPOINT_SCHEMA = "feverslop.h3-prompt-checkpoint.v1"
H3CheckpointStatus = Literal["good", "advisory_bad", "unjudged"]


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
    provenance = generated.get("prompt_provenance")
    compiler_version = (
        int(provenance.get("compiler_version") or 0)
        if isinstance(provenance, Mapping)
        else 0
    )
    if compiler_version >= 8:
        contract = generated.get("prompt_contract")
        if not isinstance(contract, Mapping):
            return "unjudged"
        if contract.get("valid") is not True:
            return "unjudged"
        if int(contract.get("compiler_version") or 0) != compiler_version:
            return "unjudged"
        expected_hash = "sha256:" + hashlib.sha256(
            str(generated.get("prompt") or "").encode("utf-8"),
        ).hexdigest()
        if contract.get("prompt_sha256") != expected_hash:
            return "unjudged"
    judge = generated.get("prompt_judge")
    verdict = str(judge.get("verdict") or "").strip().lower() if isinstance(judge, Mapping) else ""
    if verdict == "good":
        return "good"
    if verdict == "bad":
        return "advisory_bad"
    return "unjudged"


def valid_h3_prompt_contract(generated: Mapping[str, Any]) -> bool:
    provenance = generated.get("prompt_provenance")
    contract = generated.get("prompt_contract")
    if not isinstance(provenance, Mapping) or not isinstance(contract, Mapping):
        return False
    compiler_version = int(provenance.get("compiler_version") or 0)
    expected_hash = "sha256:" + hashlib.sha256(
        str(generated.get("prompt") or "").encode("utf-8"),
    ).hexdigest()
    return (
        compiler_version >= 8
        and contract.get("valid") is True
        and int(contract.get("compiler_version") or 0) == compiler_version
        and contract.get("prompt_sha256") == expected_hash
    )
