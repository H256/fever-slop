"""Pure decisions about imperfect vocal and transcript evidence."""
from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Mapping


@dataclass(frozen=True)
class TranscriptDecision:
    status: str
    reason_codes: tuple[str, ...]


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def decide_transcript(segment: Mapping[str, object]) -> TranscriptDecision:
    start, end = _finite(segment.get("start")), _finite(segment.get("end"))
    if start is None or end is None or start < 0 or end <= start:
        return TranscriptDecision("rejected", ("invalid_bounds",))
    text = str(segment.get("text") or "").strip()
    if not text:
        return TranscriptDecision("rejected", ("empty_text",))
    if "untertitelung des zdf" in text.lower():
        return TranscriptDecision("rejected", ("subtitle_hallucination",))
    no_speech = _finite(segment.get("no_speech_prob"))
    quality = _finite(segment.get("avg_logprob"))
    if quality is not None and quality >= -0.5:
        reason = "quality_overrides_no_speech" if no_speech is not None and no_speech > .85 else "good_transcript_quality"
        return TranscriptDecision("accepted", (reason,))
    reasons = []
    if no_speech is None or quality is None:
        reasons.append("missing_quality_metrics")
    if quality is not None:
        reasons.append("low_transcript_quality")
    if no_speech is not None and no_speech > .85:
        reasons.append("high_no_speech")
    return TranscriptDecision("uncertain", tuple(reasons))


@dataclass(frozen=True)
class VocalEvidence:
    activity_status: str
    transcript_status: str
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.activity_status not in {"confirmed", "uncertain", "conflict"}:
            raise ValueError("Invalid vocal activity evidence status")
        if self.transcript_status not in {"accepted", "uncertain", "rejected", "missing"}:
            raise ValueError("Invalid transcript evidence status")
        if not isinstance(self.reason_codes, tuple) or any(not isinstance(reason, str) for reason in self.reason_codes):
            raise ValueError("Evidence reason codes must be a tuple of strings")

    @classmethod
    def from_dict(cls, value: Mapping[str, object] | None) -> VocalEvidence | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError("Vocal evidence must be an object")
        reasons = value.get("reason_codes", ())
        if not isinstance(reasons, (list, tuple)):
            raise ValueError("Evidence reason codes must be an array")
        return cls(value.get("activity_status"), value.get("transcript_status"), tuple(reasons))


def merge_evidence(left: VocalEvidence | None, right: VocalEvidence | None) -> VocalEvidence | None:
    if left is None and right is None:
        return None
    unknown = VocalEvidence("uncertain", "uncertain", ("legacy_evidence_unknown",))
    left = left or unknown
    right = right or unknown
    activity = left.activity_status if left.activity_status == right.activity_status else "uncertain"
    if "conflict" in (left.activity_status, right.activity_status):
        activity = "conflict"
    transcript = left.transcript_status if left.transcript_status == right.transcript_status else "uncertain"
    return VocalEvidence(activity, transcript, tuple(dict.fromkeys(left.reason_codes + right.reason_codes)))
