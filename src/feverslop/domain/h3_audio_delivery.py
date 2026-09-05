"""Semantic audio-delivery contract declared by MiniMax H3 workflow profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


_AUDIO_LATENT_POLICIES = frozenset({"preserve_original_av_audio_latent"})


@dataclass(frozen=True)
class H3AudioDelivery:
    """How a selected H3 workflow carries supplied audio into a render.

    This is deliberately derived from the workflow sidecar rather than audio
    filenames: a full mix can be a generation condition, an output copy, an
    audience-only score, or none of these depending on the workflow graph.
    """

    audio_policy: str = "not_applicable"
    conditions_generation: bool = False
    copies_to_output: bool = False
    is_audience_only_music: bool = False
    workflow_profile: str | None = None

    def to_context(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_context(cls, value: object) -> "H3AudioDelivery":
        if not isinstance(value, dict):
            return cls()
        return cls(
            audio_policy=str(value.get("audio_policy") or "not_applicable"),
            conditions_generation=value.get("conditions_generation") is True,
            copies_to_output=value.get("copies_to_output") is True,
            is_audience_only_music=value.get("is_audience_only_music") is True,
            workflow_profile=(
                str(value["workflow_profile"])
                if value.get("workflow_profile") else None
            ),
        )


def load_h3_audio_delivery(workflow_path: str | Path | None) -> H3AudioDelivery:
    """Read a workflow's optional ``.profile.json`` audio delivery semantics.

    Missing or malformed sidecars never invent audio behavior. The renderer
    remains the authority for whether an audio reference is attached.
    """
    if workflow_path is None:
        return H3AudioDelivery()
    path = Path(workflow_path)
    profile_path = path.with_suffix(".profile.json")
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return H3AudioDelivery()
    if not isinstance(raw, dict):
        return H3AudioDelivery()
    policy = str(raw.get("audio_policy") or "not_applicable").strip().casefold()
    preserves_audio_latent = raw.get("preserve_audio_latent") is True
    has_audio_latent_topology = "#RECOMBINE_AV" in {
        str(item).strip().upper() for item in raw.get("topology") or ()
    }
    conditions_generation = (
        policy in _AUDIO_LATENT_POLICIES
        and preserves_audio_latent
        and has_audio_latent_topology
    )
    return H3AudioDelivery(
        audio_policy=policy,
        conditions_generation=conditions_generation,
        copies_to_output=conditions_generation,
        is_audience_only_music=False,
        workflow_profile=str(profile_path),
    )
