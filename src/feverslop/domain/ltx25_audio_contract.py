"""Semantic audio-delivery contract declared by LTX 2.5 workflow profiles.

Mirrors :mod:`feverslop.domain.h3_audio_delivery` for the LTX 2.5 family. The
``.profile.json`` sidecar next to a workflow is the source of truth for the
declared audio/timing policy; the workflow graph is the source of truth for the
actual topology. ``validate_ltx25_audio_workflow`` rejects the two kinds of
unsupported combinations the issue calls out:

* a declared policy the graph cannot satisfy (e.g. ``native_audio`` declared but
  no audio source / no audio-latent two-pass chain), and
* a declared policy that contradicts the graph (e.g. ``preserve_audio_latent``
  declared but the second pass drops the audio latent).

Missing or malformed sidecars never invent behavior; they default to a policy
that is always satisfiable so a workflow without a sidecar still renders.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

NATIVE_AUDIO_POLICY = "native_audio_when_declared"
ABSOLUTE_TIMING_POLICY = "absolute_frame_windows"
SUPPORTED_AUDIO_POLICIES = frozenset({NATIVE_AUDIO_POLICY, "not_applicable"})
SUPPORTED_TIMING_POLICIES = frozenset({ABSOLUTE_TIMING_POLICY, "not_applicable"})


class LTX25AudioContractError(ValueError):
    """A declared LTX 2.5 audio/timing contract cannot be satisfied by the graph."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class LTX25AudioPolicy:
    """Declared audio/timing contract for one LTX 2.5 workflow."""

    audio_policy: str = "not_applicable"
    timing_policy: str = "not_applicable"
    preserve_audio_latent: bool = False
    workflow_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_ltx25_audio_policy(workflow_path: str | Path | None) -> LTX25AudioPolicy:
    """Read a workflow's optional ``.profile.json`` audio/timing contract.

    A missing or malformed sidecar yields a ``not_applicable`` policy (always
    satisfiable) rather than inventing behavior, matching the H3 convention.
    """
    if workflow_path is None:
        return LTX25AudioPolicy()
    path = Path(workflow_path)
    profile_path = path.with_suffix(".profile.json")
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return LTX25AudioPolicy(workflow_path=str(path))
    if not isinstance(raw, dict):
        return LTX25AudioPolicy(workflow_path=str(path))
    return LTX25AudioPolicy(
        workflow_path=str(path),
        audio_policy=str(raw.get("audio_policy") or "not_applicable").strip().casefold(),
        timing_policy=str(raw.get("timing_policy") or "not_applicable").strip().casefold(),
        preserve_audio_latent=raw.get("preserve_audio_latent") is True,
    )


def _src(edge: object) -> str | None:
    return str(edge[0]) if isinstance(edge, list) and len(edge) == 2 else None


def _class_type(workflow: dict, node_id: str | None) -> str | None:
    node = workflow.get(node_id)
    return node.get("class_type") if isinstance(node, dict) else None


def _follow(workflow: dict, node_id: str | None, _depth: int = 0) -> str | None:
    """Resolve a node through VRAMCleanup passthroughs to a real node."""
    seen: set[str] = set()
    while _depth < 8 and node_id and _class_type(workflow, node_id) == "VRAMCleanup" and node_id not in seen:
        seen.add(node_id)
        for value in workflow[node_id].get("inputs", {}).values():
            if isinstance(value, list) and len(value) == 2:
                node_id = _src(value)
                break
        _depth += 1
    return node_id


def _has_audio_latent_two_pass_chain(workflow: dict) -> bool:
    """True when the audio latent survives both refinement passes.

    The accepted topology is:
    ``... -> LTXVSeparateAVLatent(av_latent <- SamplerCustomAdvanced(latent_image <-
    LTXVConcatAVLatent(audio_latent <- LTXVSeparateAVLatent))) ->
    LTXVAudioVAEDecode``. The second ``LTXVSeparateAVLatent`` is the audio
    latent re-extracted after the second pass; its presence is what preserves
    the audio latent through two-pass refinement.
    """
    for decode_id, node in workflow.items():
        if node.get("class_type") != "LTXVAudioVAEDecode":
            continue
        separate_id = _follow(workflow, _src(node.get("inputs", {}).get("samples")))
        if _class_type(workflow, separate_id) != "LTXVSeparateAVLatent":
            continue
        sampler_id = _follow(workflow, _src(workflow[separate_id].get("inputs", {}).get("av_latent")))
        if _class_type(workflow, sampler_id) != "SamplerCustomAdvanced":
            continue
        concat_id = _src(workflow[sampler_id].get("inputs", {}).get("latent_image"))
        if _class_type(workflow, concat_id) != "LTXVConcatAVLatent":
            continue
        audio_latent_id = _src(workflow[concat_id].get("inputs", {}).get("audio_latent"))
        if _class_type(workflow, audio_latent_id) == "LTXVSeparateAVLatent":
            return True
    return False


def _has_audio_source(workflow: dict) -> bool:
    """True when the workflow produces an audio latent (file or empty)."""
    for node in workflow.values():
        if node.get("class_type") in {"LTXVAudioVAEEncode", "LTXVEmptyLatentAudio"}:
            return True
    return False


def _has_output_audio(workflow: dict) -> bool:
    """True when a video-combine node is wired to an audio decode."""
    for node in workflow.values():
        if node.get("class_type") not in {"VHS_VideoCombine", "CreateVideo"}:
            continue
        audio_id = _src(node.get("inputs", {}).get("audio"))
        if _class_type(workflow, audio_id) == "LTXVAudioVAEDecode":
            return True
    return False


def validate_ltx25_audio_workflow(workflow: dict, policy: LTX25AudioPolicy) -> None:
    """Reject unsupported/contradictory LTX 2.5 audio combinations.

    Raises :class:`LTX25AudioContractError` with a stable ``code`` and an
    actionable English message. A ``not_applicable`` policy is always
    satisfiable (nothing to enforce), which is what keeps sidecar-less
    workflows rendering.
    """
    if not isinstance(workflow, dict):
        raise LTX25AudioContractError("ltx25_audio_invalid_payload", "Workflow payload must be an object.")
    if policy.audio_policy not in SUPPORTED_AUDIO_POLICIES:
        raise LTX25AudioContractError(
            "ltx25_audio_unsupported_policy",
            f"Unsupported audio_policy {policy.audio_policy!r}; expected one of "
            f"{', '.join(sorted(SUPPORTED_AUDIO_POLICIES))}.",
        )
    if policy.timing_policy not in SUPPORTED_TIMING_POLICIES:
        raise LTX25AudioContractError(
            "ltx25_timing_unsupported_policy",
            f"Unsupported timing_policy {policy.timing_policy!r}; expected one of "
            f"{', '.join(sorted(SUPPORTED_TIMING_POLICIES))}.",
        )
    if policy.audio_policy == "not_applicable":
        return
    if not _has_audio_source(workflow):
        raise LTX25AudioContractError(
            "ltx25_audio_missing_source",
            "Declared native audio but the workflow has no audio source "
            "(neither LTXVAudioVAEEncode nor LTXVEmptyLatentAudio).",
        )
    if policy.preserve_audio_latent and not _has_audio_latent_two_pass_chain(workflow):
        raise LTX25AudioContractError(
            "ltx25_audio_latent_not_preserved",
            "Declared preserve_audio_latent but the audio latent is not carried "
            "through the two-pass refinement chain.",
        )
    if not _has_output_audio(workflow):
        raise LTX25AudioContractError(
            "ltx25_audio_missing_output",
            "Declared native audio but no video-combine node is wired to an "
            "LTXVAudioVAEDecode output.",
        )
