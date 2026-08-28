from __future__ import annotations

import math
import re
from dataclasses import dataclass
from collections.abc import Iterable, Mapping, Sequence

from feverslop.domain.locked_scene_facts import LockedSceneFacts
from feverslop.prompting.dspy_h3_models import CreativeShotPayload, ResolvedPromptPlan

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


def validate_h3_prompt_shape(prompt: str, *, mode: str) -> list[PromptContractIssue]:
    """Validate the guide-level field shape of the final H3 prompt."""
    text = str(prompt or "")
    normalized = str(mode).strip().lower()
    if normalized in {"r2v", "ref", "reference"}:
        expected = (
            "subject_definitions:",
            "summary:",
            "retention_analysis:",
            "detailed_description:",
            "overall_soundscape:",
            "non_diegetic_music:",
        )
    else:
        expected = (
            "integrated_multimodal_description:",
            "overall_soundscape:",
            "non_diegetic_music:",
        )
    positions = [
        [match.start() for match in re.finditer(rf"(?m)^{re.escape(field)}", text)]
        for field in expected
    ]
    if any(len(matches) == 0 for matches in positions):
        return [PromptContractIssue(
            "h3.sections.missing",
            "prompt",
            "compiled prompt does not match the required H3 field structure",
        )]
    if any(len(matches) != 1 for matches in positions):
        return [PromptContractIssue(
            "h3.sections.duplicate",
            "prompt",
            "compiled prompt contains duplicate H3 field headers",
        )]
    flat_positions = [matches[0] for matches in positions]
    if flat_positions != sorted(flat_positions):
        return [PromptContractIssue(
            "h3.sections.order",
            "prompt",
            "compiled prompt fields are out of order",
        )]
    return []


def validate_h3_prompt_contract(
    prompt: str,
    *,
    mode: str,
    plan: ResolvedPromptPlan,
    reference_metadata: Sequence[Mapping[str, object]] | None = None,
    duration_seconds: float | None = None,
) -> list[PromptContractIssue]:
    """Validate compiler-owned MiniMax H3 grammar against its structured plan."""
    issues = validate_h3_prompt_shape(prompt, mode=mode)
    if issues:
        return issues
    if not isinstance(plan, ResolvedPromptPlan):
        raise TypeError("plan must be a ResolvedPromptPlan")
    normalized = str(mode).strip().lower()
    if normalized in {"r2v", "ref", "reference"}:
        issues.extend(_validate_r2v_contract(prompt, plan, reference_metadata or ()))
    else:
        issues.extend(_validate_base_contract(
            prompt, normalized, plan, duration_seconds=duration_seconds,
        ))
    return issues


def _validate_r2v_contract(
    text: str,
    plan: ResolvedPromptPlan,
    reference_metadata: Sequence[Mapping[str, object]],
) -> list[PromptContractIssue]:
    issues: list[PromptContractIssue] = []
    definitions = _h3_section(text, "subject_definitions:", "summary:")
    summary = _h3_section(text, "summary:", "retention_analysis:")
    retention = _h3_section(text, "retention_analysis:", "detailed_description:")
    detailed = _h3_section(text, "detailed_description:", "overall_soundscape:")
    style_opening = str(plan.style_opening or "").strip()
    before_first_shot = detailed.partition("[Shot 1]")[0].strip()
    if not style_opening or before_first_shot != style_opening:
        issues.append(PromptContractIssue(
            "h3.detail.style_opening",
            "detailed_description",
            "R2V detailed_description must begin with the LLM-authored style_opening before Shot 1",
        ))
    if not re.match(r"^\[[a-z][a-z +]+\]\s+\S", summary):
        issues.append(PromptContractIssue(
            "h3.summary.task_type", "summary", "summary must begin with one task-type prefix",
        ))
    word_count = len(re.findall(r"\b[\w'-]+\b", detailed))
    task_prefix = summary.partition("]")[0].casefold()
    if "video editing" not in task_prefix and word_count < 350:
        issues.append(PromptContractIssue(
            "h3.detail.too_short",
            "detailed_description",
            "generation-mode detailed_description must contain at least 350 English words "
            f"(actual: {word_count} words)",
        ))
    issues.extend(_validate_shots(detailed, plan))
    if re.search(r"\(S\d+(?:,S\d+)*\)", retention):
        issues.append(PromptContractIssue(
            "h3.retention.speaker", "retention_analysis", "speaker IDs are forbidden in retention_analysis",
        ))
    for subject in plan.subjects:
        definition_line = _line_for_label(definitions, subject.label)
        retention_line = _line_for_label(retention, subject.label)
        if definition_line is None:
            issues.append(PromptContractIssue(
                "h3.subject.definition_missing", "subject_definitions", "subject has no definition", subject.label,
            ))
        else:
            for source in subject.source_references:
                if source not in definition_line:
                    issues.append(PromptContractIssue(
                        "h3.subject.source_missing",
                        "subject_definitions",
                        "subject definition does not cite its source reference",
                        subject.label,
                    ))
        if retention_line is None:
            issues.append(PromptContractIssue(
                "h3.subject.retention_missing", "retention_analysis", "subject has no retention entry", subject.label,
            ))
        alias = re.sub(r"^(?:the\s+)", "", subject.name.strip(), flags=re.IGNORECASE)
        if alias and re.search(rf"(?<![\w>])(?:the\s+)?{re.escape(alias)}\b", detailed, re.IGNORECASE):
            issues.append(PromptContractIssue(
                "h3.subject.alias",
                "detailed_description",
                "subject name remains where the stable subject label must be used",
                subject.label,
            ))
    metadata_by_label = {
        str(item.get("label")): item
        for item in reference_metadata
        if str(item.get("label") or "").strip()
    }
    audio_labels = {
        *re.findall(r"<Audio\s+[1-9][0-9]*>", definitions),
        *(
            label for label, metadata in metadata_by_label.items()
            if str(metadata.get("kind") or "").casefold() == "audio"
        ),
    }
    for label in sorted(audio_labels):
        if label not in summary:
            issues.append(PromptContractIssue(
                "h3.audio.summary_missing",
                "summary",
                "summary does not name the audio reference relationship",
                label,
            ))
        if _line_for_label(definitions, label) is None:
            issues.append(PromptContractIssue(
                "h3.audio.definition_missing", "subject_definitions", "audio reference has no definition", label,
            ))
        if _line_for_label(retention, label) is None:
            issues.append(PromptContractIssue(
                "h3.audio.retention_missing", "retention_analysis", "audio reference has no retention entry", label,
            ))
        locations = [match.start() for match in re.finditer(re.escape(label), detailed)]
        if not locations:
            issues.append(PromptContractIssue(
                "h3.audio.missing",
                "detailed_description",
                "defined audio reference is not cited where its relationship applies",
                label,
            ))
        elif not any(
            re.search(
                r"\b(?:copied|referenced|reused)\b",
                detailed[max(0, location - 120):location + len(label) + 180],
                re.IGNORECASE,
            )
            for location in locations
        ):
            issues.append(PromptContractIssue(
                "h3.audio.relationship_missing",
                "detailed_description",
                "audio reference citation must state whether its signal is copied or referenced",
                label,
            ))
    issues.extend(_validate_dialogue(detailed))
    return issues


def _validate_base_contract(
    text: str,
    mode: str,
    plan: ResolvedPromptPlan,
    *,
    duration_seconds: float | None,
) -> list[PromptContractIssue]:
    issues: list[PromptContractIssue] = []
    detailed = _h3_section(text, "integrated_multimodal_description:", "overall_soundscape:")
    issues.extend(_validate_shots(detailed, plan))
    duration = float(duration_seconds or 0.0)
    final_shot = max((shot.shot_number for shot in plan.shots), default=1)
    instruction = text.split("\n\n", 1)[0]
    expected = None
    if mode == "i2v":
        expected = (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced."
        )
    elif mode == "fl2v":
        expected = (
            "How the reference pictures align with the target video — "
            "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
            f"Picture 2 (from Shot {final_shot}) aligns with the {duration:.2f}-second mark of the target video."
        )
    elif mode == "l2v":
        expected = (
            "How the reference pictures align with the target video — "
            f"<Picture 1> (from [Shot {final_shot}]) aligns with the {duration:.2f}-second mark of the target video."
        )
    if expected is not None and instruction != expected:
        issues.append(PromptContractIssue(
            "h3.alignment.formula", "alignment_instruction", "frame alignment instruction does not match the guide formula",
        ))
    required_labels = {
        "i2v": ("<Picture 1>",),
        "fl2v": ("<Picture 1>", "<Picture 2>"),
        "l2v": ("<Picture 1>",),
    }.get(mode, ())
    for label in required_labels:
        if label not in detailed:
            issues.append(PromptContractIssue(
                "h3.frame_anchor.missing", "integrated_multimodal_description", "frame anchor is absent from the timeline", label,
            ))
    issues.extend(_validate_dialogue(detailed))
    return issues


def _validate_shots(text: str, plan: ResolvedPromptPlan) -> list[PromptContractIssue]:
    issues: list[PromptContractIssue] = []
    markers = list(re.finditer(
        r"(?m)^\[Shot\s+(\d+)\](?:\s+At\s+([0-9]{2}:[0-9]{2}\.[0-9]{3}),)?",
        text,
    ))
    expected_numbers = [shot.shot_number for shot in plan.shots]
    actual_numbers = [int(marker.group(1)) for marker in markers]
    if actual_numbers != expected_numbers:
        issues.append(PromptContractIssue(
            "h3.shot.sequence", "detailed_description", "shot labels must occur exactly once in planned order",
        ))
        return issues
    for index, (marker, shot) in enumerate(zip(markers, plan.shots, strict=True)):
        timestamp = marker.group(2)
        if index == 0 and timestamp is not None:
            issues.append(PromptContractIssue(
                "h3.shot.first_timestamp", f"shots[{index}]", "Shot 1 must not have a timestamp",
            ))
        if index > 0:
            expected = _h3_time(float(shot.start_seconds or 0.0))
            if timestamp != expected:
                issues.append(PromptContractIssue(
                    "h3.shot.timestamp", f"shots[{index}]", f"shot cut timestamp must be {expected}",
                ))
    return issues


def _validate_dialogue(text: str) -> list[PromptContractIssue]:
    issues: list[PromptContractIssue] = []
    if text.count("<d>") != text.count("</d>"):
        return [PromptContractIssue(
            "h3.dialogue.unbalanced", "detailed_description", "dialogue tags are unbalanced",
        )]
    for match in re.finditer(r"<d>(.*?)</d>", text, re.DOTALL | re.IGNORECASE):
        content = match.group(1).strip()
        if not re.match(r"^\[[^]\r\n]+\]\s+\S", content):
            issues.append(PromptContractIssue(
                "h3.dialogue.language", "detailed_description", "dialogue must begin with a language tag",
            ))
        spoken = re.sub(r"^\[[^]]+\]\s*", "", content).rstrip()
        if spoken and spoken[-1] not in ".?!>":
            issues.append(PromptContractIssue(
                "h3.dialogue.punctuation", "detailed_description", "complete dialogue must end with punctuation before </d>",
            ))
    return issues


def _line_for_label(section: str, label: str) -> str | None:
    return next(
        (line.strip() for line in section.splitlines() if line.strip().startswith(label)),
        None,
    )


def _h3_time(seconds: float) -> str:
    milliseconds = int(round(float(seconds) * 1000))
    minutes, remainder = divmod(milliseconds, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def _h3_section(text: str, header: str, next_header: str) -> str:
    start = text.index(header) + len(header)
    end = text.index(next_header, start)
    return text[start:end].strip()


__all__ = [
    "PromptContractError",
    "PromptContractIssue",
    "validate_h3_prompt_contract",
    "validate_h3_prompt_shape",
    "validate_prompt_contract",
]
