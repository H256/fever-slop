from __future__ import annotations

import math
import re
from dataclasses import dataclass
from collections.abc import Iterable, Mapping, Sequence

from feverslop.domain.locked_scene_facts import LockedSceneFacts
from feverslop.prompting.dspy_h3_models import CreativeShotPayload

_REFERENCE_LABEL = re.compile(r"<(?:Picture|Audio|Video) [1-9][0-9]*>")


@dataclass(frozen=True)
class PromptContractIssue:
    """A deterministic, non-sensitive structural prompt diagnostic."""

    code: str
    path: str
    message: str
    source_id: str | None = None


class PromptContractError(ValueError):
    """Raised when deterministic prompt validation blocks preparation."""

    def __init__(self, issues: Sequence[PromptContractIssue]):
        self.issues = tuple(issues)
        super().__init__("prompt contract validation failed: " + "; ".join(issue.code for issue in self.issues))


def validate_prompt_contract(
    prompt: str,
    *,
    facts: LockedSceneFacts,
    shots: Sequence[CreativeShotPayload],
    shot_windows: Mapping[str, tuple[float, float]],
    references: Mapping[str, Sequence[str]] | None = None,
    prepared_reference_labels: Iterable[str] | None = None,
    duration_seconds: float | None = None,
) -> list[PromptContractIssue]:
    """Validate compiled prompt coverage and structure without leaking content."""
    issues: list[PromptContractIssue] = []
    text = str(prompt or "")
    if not isinstance(facts, LockedSceneFacts):
        raise TypeError("facts must be LockedSceneFacts")

    for index, fact in enumerate(facts.facts):
        if fact.value not in text:
            issues.append(PromptContractIssue(
                "fact.missing",
                f"facts[{index}].value",
                "locked fact value is not represented in the compiled prompt",
                fact.source_id,
            ))

    expected_labels: set[str] = set()
    for shot_index, shot in enumerate(shots):
        if not isinstance(shot, CreativeShotPayload):
            raise TypeError("shots must contain CreativeShotPayload values")
        expected_labels.update(
            str(label).strip()
            for label in (references or {}).get(shot.shot_id, ())
            if str(label).strip()
        )
        labels = [str(label).strip() for label in (references or {}).get(shot.shot_id, ()) if str(label).strip()]
        if len(labels) != len(set(labels)):
            issues.append(PromptContractIssue(
                "reference.duplicate",
                f"references.{shot.shot_id}",
                "shot contains duplicate reference labels",
                shot.shot_id,
            ))
        marker = f"[Shot {shot_index + 1} |"
        if marker not in text:
            issues.append(PromptContractIssue(
                "shot.missing",
                f"shots[{shot_index}]",
                "planned shot is not represented in the compiled prompt",
                shot.shot_id,
            ))

    prepared = {str(label).strip() for label in (prepared_reference_labels or ()) if str(label).strip()}
    labels_in_prompt = set(_REFERENCE_LABEL.findall(text))
    for label in sorted(expected_labels):
        if prepared and label not in prepared:
            issues.append(PromptContractIssue(
                "reference.unknown",
                "references",
                "reference label is not present in prepared input slots",
                label,
            ))
        elif label not in labels_in_prompt:
            issues.append(PromptContractIssue(
                "reference.missing",
                "references",
                "planned reference label is not represented in the compiled prompt",
                label,
            ))
    for label in sorted(labels_in_prompt - expected_labels):
        if not prepared or label not in prepared:
            issues.append(PromptContractIssue(
                "reference.unknown",
                "prompt.references",
                "prompt contains a reference label outside prepared input slots",
                label,
            ))

    previous_end: float | None = None
    for index, shot in enumerate(shots):
        window = shot_windows.get(shot.shot_id)
        if window is None:
            issues.append(PromptContractIssue(
                "timing.missing",
                f"shot_windows.{shot.shot_id}",
                "planned shot has no timing window",
                shot.shot_id,
            ))
            continue
        start, end = (float(window[0]), float(window[1]))
        path = f"shot_windows.{shot.shot_id}"
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            issues.append(PromptContractIssue("timing.invalid", path, "timing window is invalid", shot.shot_id))
            continue
        if previous_end is not None and start < previous_end:
            issues.append(PromptContractIssue("timing.overlap", path, "timing windows overlap or are out of order", shot.shot_id))
        previous_end = end
        if duration_seconds is not None and end > float(duration_seconds):
            issues.append(PromptContractIssue("timing.duration_exceeded", path, "timing window exceeds scene duration", shot.shot_id))

    return issues


__all__ = ["PromptContractError", "PromptContractIssue", "validate_prompt_contract"]
