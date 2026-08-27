from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from feverslop.domain.locked_scene_facts import LockedSceneFacts
from feverslop.prompting.dspy_h3_models import CreativeShotPayload
from feverslop.prompting.dspy_h3_models import PromptMode
from feverslop.prompting.dspy_h3_models import ResolvedPromptPlan
from feverslop.prompting.prompt_contract_validation import PromptContractError, validate_prompt_contract


H3_COMPILER_NAME = "deterministic_h3_compiler"
H3_COMPILER_VERSION = 8


class DeterministicH3Compiler:
    """Compile structured facts and creative fields without model or I/O side effects."""

    def __init__(self, *, max_words: int | None = None) -> None:
        if max_words is not None and (isinstance(max_words, bool) or max_words <= 0):
            raise ValueError("max_words must be positive or None")
        self.max_words = max_words

    def compile(
        self,
        *,
        mode: str,
        plan: ResolvedPromptPlan | None = None,
        facts: LockedSceneFacts,
        shots: Sequence[CreativeShotPayload],
        shot_windows: Mapping[str, tuple[float, float]],
        references: Mapping[str, Sequence[str]] | None = None,
        prepared_reference_labels: Sequence[str] | None = None,
        reference_metadata: Sequence[Mapping[str, Any]] | None = None,
        duration_seconds: float | None = None,
        dialogue_language: str | None = None,
    ) -> str:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"base", "reference", "ref", *(item.value for item in PromptMode)}:
            raise ValueError("mode must be base, reference, or a PromptMode value")
        if plan is not None:
            return self._compile_guide_prompt(
                mode=(
                    PromptMode.R2V
                    if normalized_mode in {"reference", "ref", "r2v"}
                    else PromptMode.T2V
                    if normalized_mode == "base"
                    else PromptMode(normalized_mode)
                ),
                plan=plan,
                facts=facts,
                references=references,
                prepared_reference_labels=prepared_reference_labels,
                reference_metadata=reference_metadata,
                duration_seconds=duration_seconds,
                dialogue_language=dialogue_language,
            )
        if not isinstance(facts, LockedSceneFacts):
            raise TypeError("facts must be LockedSceneFacts")
        by_id: dict[str, CreativeShotPayload] = {}
        for shot in shots:
            if not isinstance(shot, CreativeShotPayload):
                raise TypeError("shots must contain CreativeShotPayload values")
            if shot.shot_id in by_id:
                raise ValueError(f"duplicate shot ID: {shot.shot_id}")
            if shot.shot_id not in shot_windows:
                raise ValueError(f"missing timing window for shot: {shot.shot_id}")
            by_id[shot.shot_id] = shot

        lines = ["BASE PROMPT" if normalized_mode == "base" else "FULL REFERENCE PROMPT", f"Scene: {facts.scene_id}"]
        if facts.facts:
            lines.append("Locked facts:")
            lines.extend(f"- {fact.category}/{fact.key}: {fact.value}" for fact in facts.facts)
        for index, shot_id in enumerate(sorted(by_id), start=1):
            start, end = shot_windows[shot_id]
            if float(start) < 0 or float(end) <= float(start):
                raise ValueError(f"invalid timing window for shot: {shot_id}")
            shot = by_id[shot_id]
            lines.append(f"[Shot {index} | {_time(start)}-{_time(end)}]")
            lines.append(f"Action: {shot.visible_action.strip()}")
            lines.append(f"Performance: {shot.performance.strip()}")
            if shot.camera_behavior:
                lines.append(f"Camera: {shot.camera_behavior.strip()}")
            if shot.environmental_motion:
                lines.append(f"Environment motion: {shot.environmental_motion.strip()}")
            if shot.transition_intent:
                lines.append(f"Transition: {shot.transition_intent.strip()}")
            labels = sorted({str(label).strip() for label in (references or {}).get(shot_id, ()) if str(label).strip()})
            if labels:
                lines.append("References: " + ", ".join(labels))
        result = "\n".join(lines)
        if self.max_words is not None and len(result.split()) > self.max_words:
            raise ValueError(f"compiled prompt exceeds word budget ({self.max_words})")
        issues = validate_prompt_contract(
            result,
            facts=facts,
            shots=tuple(by_id[key] for key in sorted(by_id)),
            shot_windows=shot_windows,
            references=references,
            prepared_reference_labels=prepared_reference_labels,
            duration_seconds=duration_seconds,
        )
        if issues:
            raise PromptContractError(issues)
        return result

    def _compile_guide_prompt(
        self,
        *,
        mode: PromptMode,
        plan: ResolvedPromptPlan,
        facts: LockedSceneFacts,
        references: Mapping[str, Sequence[str]] | None,
        prepared_reference_labels: Sequence[str] | None,
        reference_metadata: Sequence[Mapping[str, Any]] | None,
        duration_seconds: float | None,
        dialogue_language: str | None,
    ) -> str:
        """Serialize LLM-authored fields into the MiniMax guide grammar."""
        if mode is PromptMode.R2V:
            subject_lines = [
                _render_subject_definition(subject)
                for subject in plan.subjects
            ]
            metadata_by_label = {
                str(reference.get("label")): reference
                for reference in reference_metadata or ()
                if str(reference.get("label") or "").strip()
            }
            represented = {
                *[label for subject in plan.subjects for label in subject.source_references],
                *[usage.reference_label for usage in plan.reference_usage],
            }
            for label in prepared_reference_labels or ():
                if label in represented:
                    continue
                subject_lines.append(_render_prepared_reference_definition(
                    label, metadata_by_label.get(label, {}),
                ))
            subject_sources = {
                label for subject in plan.subjects for label in subject.source_references
            }
            for usage in plan.reference_usage:
                if usage.reference_label.lower().startswith("<audio "):
                    metadata = metadata_by_label.get(usage.reference_label, {})
                    copy_mode = str(metadata.get("copy_mode") or "reference")
                    name = str(metadata.get("name") or "audio reference").strip()
                    role = {
                        "full_mix": "complete soundtrack",
                        "vocals": "vocal stem",
                    }.get(name.casefold(), name.replace("_", " "))
                    relationship = {
                        "fully_copy": "is fully copied as the target video's audio signal",
                        "partially_copy": "is partially copied into the target video's audio signal",
                    }.get(copy_mode, "is referenced for the target video's soundtrack")
                    subject_lines.append(
                        f"{usage.reference_label} is the {role} and {relationship}."
                    )
                elif usage.reference_label not in subject_sources:
                    subject_lines.append(
                        f"{usage.reference_label} is the reference input for {usage.purpose}; "
                        f"{usage.details.rstrip('.')} .".replace(" .", ".")
                    )
            frame_roles = {"first_frame", "last_frame", "keyframe", "storyboard", "composition"}
            retention_lines = [
                f"{subject.label} (appears in "
                f"{', '.join(f'[Shot {number}]' for number in _subject_shot_numbers(subject, plan))}): "
                f"fully_preserved - {_clean_subject_description(subject)}"
                for subject in plan.subjects
            ]
            usage_retention_lines: list[str] = []
            for usage in plan.reference_usage:
                reference = metadata_by_label.get(usage.reference_label, {})
                kind = str(reference.get("kind") or "").lower()
                role = str(reference.get("role") or usage.purpose or "").lower()
                if kind == "picture" and role not in frame_roles:
                    continue
                marker = (
                    str(reference.get("copy_mode") or "reference")
                    if kind == "audio"
                    else "fully_preserved"
                )
                if marker not in {
                    "fully_preserved", "partially_preserved", "attribute_transfer",
                    "weak_reference", "fully_copy", "partially_copy", "reference",
                }:
                    marker = "reference" if kind == "audio" else "fully_preserved"
                usage_retention_lines.append(f"{usage.reference_label}: {marker} - {usage.details}")
            for label in prepared_reference_labels or ():
                if label in represented:
                    continue
                metadata = metadata_by_label.get(label, {})
                marker = (
                    str(metadata.get("copy_mode") or "reference")
                    if str(metadata.get("kind") or "").lower() == "audio"
                    else "fully_preserved"
                )
                retention_lines.append(f"{label}: {marker} - reference is applied in the target video.")
            retention_lines.extend(usage_retention_lines)
            detailed_parts = [str(plan.style_opening or "").strip()]
            audio_labels = {
                usage.reference_label
                for usage in plan.reference_usage
                if usage.reference_label.lower().startswith("<audio ")
            }
            audio_labels.update(
                label for label, metadata in metadata_by_label.items()
                if str(metadata.get("kind") or "").casefold() == "audio"
            )
            audio_relationships = {
                label: _audio_relationship_phrase(label, metadata_by_label.get(label, {}))
                for label in sorted(audio_labels)
            }
            detailed_parts.extend(
                _render_shot_with_references(
                    index,
                    shot,
                    plan,
                    audio_relationships=audio_relationships,
                    reference_metadata=metadata_by_label,
                    final_shot=index == len(plan.shots),
                )
                + (
                    " " + _replace_subject_names(
                        _render_camera_behavior(shot.camera_behavior), plan,
                    )
                    if shot.camera_behavior and not _description_covers_camera(shot)
                    else ""
                )
                for index, shot in enumerate(plan.shots, start=1)
            )
            detailed = _normalize_r2v_dialogue("\n".join(detailed_parts), dialogue_language or "English", plan)
            soundscape = _replace_subject_names(plan.overall_soundscape, plan)
            for lyric in _dialogue_contents(detailed):
                soundscape = _remove_sentence_containing(soundscape, lyric)
            soundscape = _remove_music_sentences(soundscape)
            if not soundscape.strip():
                soundscape = "No additional diegetic ambience or physical sound effects are specified."
            sections = [
                "subject_definitions:\n" + "\n".join(subject_lines),
                "summary: " + _summary_prefix(plan, metadata_by_label) + " "
                + _render_summary(plan, metadata_by_label),
                "retention_analysis:\n" + "\n".join(retention_lines),
                "detailed_description: " + detailed,
                "overall_soundscape: " + soundscape,
                "non_diegetic_music: " + (plan.non_diegetic_music or "N/A"),
            ]
        else:
            detailed = "\n".join(
                _render_base_shot(index, shot, mode, final_shot=index == len(plan.shots))
                for index, shot in enumerate(plan.shots, start=1)
            )
            sections = [
                "integrated_multimodal_description: " + detailed,
                "overall_soundscape: " + plan.overall_soundscape,
                "non_diegetic_music: " + (plan.non_diegetic_music or "N/A"),
            ]
            instruction = None
            duration = float(duration_seconds or 0.0)
            if not instruction and mode is PromptMode.I2V:
                instruction = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
            elif not instruction and mode is PromptMode.FL2V:
                instruction = (
                    "How the reference pictures align with the target video — "
                    "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
                    f"Picture 2 (from Shot {max((shot.shot_number for shot in plan.shots), default=1)}) "
                    f"aligns with the {duration:.2f}-second mark of the target video."
                )
            elif not instruction and mode is PromptMode.L2V:
                final_shot = max((shot.shot_number for shot in plan.shots), default=1)
                instruction = (
                    "How the reference pictures align with the target video — "
                    f"<Picture 1> (from [Shot {final_shot}]) aligns with the {duration:.2f}-second mark of the target video."
                )
            if instruction:
                sections.insert(0, instruction)
        result = "\n\n".join(section.strip() for section in sections)
        if self.max_words is not None and len(result.split()) > self.max_words:
            raise ValueError(f"compiled prompt exceeds word budget ({self.max_words})")
        return result


def _subject_shot_numbers(subject: Any, plan: ResolvedPromptPlan) -> tuple[int, ...]:
    numbers = tuple(
        shot.shot_number
        for shot in plan.shots
        if subject.label in _shot_reference_labels(shot, plan)
    )
    return numbers or tuple(shot.shot_number for shot in plan.shots[:1])


def _shot_reference_labels(shot: Any, plan: ResolvedPromptPlan) -> tuple[str, ...]:
    involved = {str(value).casefold() for value in getattr(shot, "involved_subjects", ())}
    subject_labels = [
        subject.label
        for subject in plan.subjects
        if (
            set(subject.source_references).intersection(shot.reference_labels)
            or subject.name.casefold() in involved
            or subject.label.casefold() in involved
            or (
                subject.name.strip()
                and re.search(rf"\b{re.escape(subject.name)}\b", str(shot.description), re.IGNORECASE)
            )
        )
    ]
    return tuple(dict.fromkeys(subject_labels))


def _render_subject_definition(subject: Any) -> str:
    description = _lower_initial(_clean_subject_description(subject))
    sources = " and ".join(subject.source_references)
    return f"{subject.label} is {description} in {sources}." if sources else f"{subject.label} is {description}."


def _clean_subject_description(subject: Any) -> str:
    description = str(subject.description).strip().rstrip(".")
    return re.sub(
        rf"^{re.escape(subject.label)}\s+is\s+", "", description, flags=re.IGNORECASE,
    )


def _render_shot_with_references(
    index: int,
    shot: Any,
    plan: ResolvedPromptPlan,
    *,
    audio_relationships: Mapping[str, str] | None = None,
    reference_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    final_shot: bool = False,
) -> str:
    labels = _shot_reference_labels(shot, plan)
    description = _replace_subject_names(str(shot.description), plan)
    missing = [label for label in labels if label not in description]
    if missing:
        visible = " and ".join(missing)
        verb = "is" if len(missing) == 1 else "are"
        description = f"{visible} {verb} visible in the shot. {description}"
    available_audio = audio_relationships or {}
    audio_labels = [
        label
        for label in shot.reference_labels
        if label in available_audio
    ]
    assigned_audio = {
        label
        for planned_shot in plan.shots
        for label in planned_shot.reference_labels
        if label in available_audio
    }
    if index == 1:
        audio_labels.extend(
            label for label in available_audio
            if label not in assigned_audio and label not in audio_labels
        )
    if audio_labels:
        relationships = " and ".join(available_audio[label] for label in audio_labels)
        description = f"{description.rstrip('.')} with {relationships}."
    subject_sources = {
        label for subject in plan.subjects for label in subject.source_references
    }
    metadata_by_label = reference_metadata or {}
    standalone_labels = [
        label for label in shot.reference_labels
        if label not in subject_sources and label not in available_audio
    ]
    for label in standalone_labels:
        role = str(metadata_by_label.get(label, {}).get("role") or "reference").casefold()
        if role == "first_frame" and index == 1:
            description = f"The shot begins from {label}, preserving its composition. {description}"
        elif role == "last_frame" and final_shot:
            description = f"{description.rstrip('.')} and ends on {label}."
        else:
            description = (
                f"{description.rstrip('.')} while {label} guides the applicable visual relationship."
            )
    cut = "" if index == 1 else f" At {_time(float(shot.start_seconds or 0.0))},"
    return f"[Shot {index}]{cut} {description}"


def _replace_subject_names(
    text: str,
    plan: ResolvedPromptPlan,
    labels: Sequence[str] | None = None,
) -> str:
    allowed = set(labels or (subject.label for subject in plan.subjects))
    replacements = sorted(
        (
            phrase,
            subject.label,
        )
        for subject in plan.subjects
        if subject.label in allowed
        for phrase in (subject.name,)
        if phrase.strip()
    )
    for phrase, label in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        normalized_phrase = re.sub(r"^(?:the\s+)", "", phrase.strip(), flags=re.IGNORECASE)
        text = re.sub(
            rf"(?<![\w>])(?:the\s+)?{re.escape(normalized_phrase)}(?=\b|'s\b)",
            label,
            text,
            flags=re.IGNORECASE,
        )
    return text


def _audio_relationship_phrase(label: str, metadata: Mapping[str, Any]) -> str:
    copy_mode = str(metadata.get("copy_mode") or "reference")
    if copy_mode == "fully_copy":
        return f"{label} fully copied as the complete soundtrack and timing reference"
    if copy_mode == "partially_copy":
        return f"the selected signal from {label} partially copied into the target video's audio"
    return f"{label} referenced for the target video's soundtrack"


def _render_prepared_reference_definition(label: str, metadata: Mapping[str, Any]) -> str:
    kind = str(metadata.get("kind") or "reference").casefold()
    if kind == "audio":
        relationship = _audio_relationship_phrase(label, metadata)
        return f"{label} is an audio input; {relationship}."
    role = str(metadata.get("role") or "reference input").replace("_", " ")
    description = str(metadata.get("description") or "its supplied visual characteristics").strip().rstrip(".")
    return f"{label} is the {role} reference, defining {description}."


def _summary_prefix(
    plan: ResolvedPromptPlan,
    metadata_by_label: Mapping[str, Mapping[str, Any]],
) -> str:
    task_types: list[str] = []
    roles = {str(item.get("role") or "").casefold() for item in metadata_by_label.values()}
    if roles.intersection({"first_frame", "last_frame", "keyframe"}):
        task_types.append("keyframe completion")
    if "edit_source" in roles:
        task_types.append("video editing")
    elif "continuation" in roles:
        task_types.append("video continuation")
    if plan.subjects or roles.difference({"audio_reuse"}):
        task_types.append("reference generation")
    copy_modes = {
        str(item.get("copy_mode") or "").casefold()
        for item in metadata_by_label.values()
        if str(item.get("kind") or "").casefold() == "audio"
    }
    if copy_modes.intersection({"fully_copy", "partially_copy"}):
        task_types.append("audio reuse")
    if "reference" in copy_modes:
        task_types.append("audio reference")
    return "[" + " + ".join(dict.fromkeys(task_types or ["reference generation"])) + "]"


def _render_summary(
    plan: ResolvedPromptPlan,
    metadata_by_label: Mapping[str, Mapping[str, Any]],
) -> str:
    summary = _replace_subject_names(plan.creative_intent, plan).strip().rstrip(".") + "."
    audio_labels = {
        usage.reference_label
        for usage in plan.reference_usage
        if usage.reference_label.casefold().startswith("<audio ")
    }
    audio_labels.update(
        label for label, metadata in metadata_by_label.items()
        if str(metadata.get("kind") or "").casefold() == "audio"
    )
    for label in sorted(audio_labels):
        if label in summary:
            continue
        copy_mode = str(metadata_by_label.get(label, {}).get("copy_mode") or "reference")
        relationship = {
            "fully_copy": f"{label} is fully copied as the complete target-video audio track.",
            "partially_copy": f"Selected signal from {label} is partially copied into the target video.",
        }.get(copy_mode, f"{label} is referenced without copying its original signal.")
        summary += " " + relationship
    return summary


def _render_base_shot(index: int, shot: Any, mode: PromptMode, *, final_shot: bool) -> str:
    description = str(shot.description).strip()
    if mode in {PromptMode.I2V, PromptMode.FL2V} and index == 1:
        description = f"The shot begins from <Picture 1>, preserving its composition. {description}"
    if mode is PromptMode.FL2V and final_shot:
        description = f"{description.rstrip('.')} and ends on <Picture 2>."
    elif mode is PromptMode.L2V and final_shot:
        description = f"{description.rstrip('.')} and converges to the final frame in <Picture 1>."
    cut = "" if index == 1 else f" At {_time(float(shot.start_seconds or 0.0))},"
    return f"[Shot {index}]{cut} {description}"


def _lower_initial(value: str) -> str:
    return value[:1].lower() + value[1:] if value else value


def _description_covers_camera(shot: Any) -> bool:
    description = str(shot.description or "").lower()
    return "camera" in description


def _render_camera_behavior(value: str) -> str:
    text = _lower_initial(str(value).strip().rstrip("."))
    replacements = {
        "tracking": "tracks",
        "zooming": "zooms",
        "tilting": "tilts",
        "panning": "pans",
        "continuing": "continues",
        "completing": "completes",
        "moving": "moves",
        "orbiting": "orbits",
    }
    match = re.match(r"(?:(slowly|quickly|gently|steadily)\s+)?(\w+)(.*)", text)
    if match and match.group(2) in replacements:
        adverb = f" {match.group(1)}" if match.group(1) else ""
        return f"The camera{adverb} {replacements[match.group(2)]}{match.group(3)}."
    return f"The camera movement is described as {text}."


def _normalize_r2v_dialogue(text: str, language: str, plan: ResolvedPromptPlan) -> str:
    """Canonicalize mechanical dialogue syntax after creative fields are authored."""
    language_value = str(language or "English").strip() or "English"
    language = {
        "en": "English",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
    }.get(language_value.casefold(), language_value)
    def tagged(match: re.Match[str]) -> str:
        content = match.group(1).strip()
        content = re.sub(r"^(?:\[[^]]+\]|en|de|fr|es)\s*", "", content, flags=re.IGNORECASE)
        content = _ensure_dialogue_punctuation(content)
        return f"<d>[{language}] {content}</d>"

    def split_tagged(match: re.Match[str]) -> str:
        return f"<d>[{language}] {_ensure_dialogue_punctuation(match.group(1).strip())}</d>"

    normalized = re.sub(
        r"<d>\s*(?:\[[^]]+\])?\s*</d>\s*(.*?)\s*<</d>",
        split_tagged,
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    normalized = re.sub(
        r"<d>\s*<d>\s*(.*?)\s*</d>\s*</d>",
        split_tagged,
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )
    normalized = re.sub(r"<d>\s*(.*?)\s*</d>", tagged, normalized, flags=re.IGNORECASE | re.DOTALL)

    def quoted(match: re.Match[str]) -> str:
        prefix, content = match.group(1), match.group(3).strip()
        content = _ensure_dialogue_punctuation(content)
        return f"{prefix}<d>[{language}] {content}</d>"

    normalized = re.sub(
        r"(?i)(\b(?:sings?|singing|says?|saying)(?:\s+the\s+(?:words?|lyrics?))?\s*)"
        r"(['\"])([^'\"\n]+)\2",
        quoted,
        normalized,
    )
    normalized = re.sub(r"</d>\s*[.?!](?=\s|$)", "</d>", normalized, flags=re.IGNORECASE)
    # Remove speaker IDs from every visible subject first. They are re-added
    # only to the first human subject in an explicit vocal clause.
    for subject in plan.subjects:
        normalized = re.sub(
            rf"({re.escape(subject.label)})\s*\(S\d+\)", r"\1", normalized, flags=re.IGNORECASE,
        )
    human_subjects = [
        subject for subject in plan.subjects
        if re.search(r"\b(?:man|woman|person|human|singer|performer|actor|actress|child)\b",
                     f"{subject.name} {subject.description}", re.IGNORECASE)
    ]
    if len(human_subjects) == 1:
        label = human_subjects[0].label
        normalized = re.sub(
            r"\b(?:He|She|They)\s+(?=(?:resumes\s+)?(?:sings?|singing|says?|speaks?)\b)",
            f"{label} ",
            normalized,
        )
    if human_subjects:
        speaker_ids: dict[str, int] = {}
        offset = 0
        for dialogue in tuple(re.finditer(r"<d>\s*\[[^]]+\]", normalized, re.IGNORECASE)):
            start = dialogue.start() + offset
            shot_start = normalized.rfind("\n[Shot ", 0, start)
            prefix = normalized[max(0, shot_start):start]
            relative, subject = max(
                ((prefix.rfind(item.label), item) for item in human_subjects),
                key=lambda item: item[0],
            )
            if relative < 0:
                continue
            label = subject.label
            anchor = max(0, shot_start) + relative + len(label)
            if normalized[anchor:anchor + 6].lstrip().startswith("(S"):
                continue
            speaker_id = speaker_ids.setdefault(label, len(speaker_ids) + 1)
            marker = f" (S{speaker_id})"
            normalized = normalized[:anchor] + marker + normalized[anchor:]
            offset += len(marker)
    return normalized


def _ensure_dialogue_punctuation(content: str) -> str:
    text = str(content).strip()
    if not text or re.search(r"<(?:scenetrans|cutoff)>\s*$", text, re.IGNORECASE):
        return text
    if text[-1] not in ".?!":
        text += "."
    return text


def _dialogue_contents(text: str) -> tuple[str, ...]:
    return tuple(match.strip() for match in re.findall(r"<d>\s*\[[^]]+\]\s*(.*?)\s*</d>", text, re.IGNORECASE | re.DOTALL) if match.strip())


def _remove_sentence_containing(text: str, phrase: str) -> str:
    if not phrase:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [part for part in parts if phrase.casefold() not in part.casefold()]
    return " ".join(kept).strip()


def _remove_music_sentences(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    music = re.compile(r"\b(?:musical track|full mix|song|vocals?|background music)\b", re.IGNORECASE)
    return " ".join(part for part in parts if not music.search(part)).strip()


def creative_shots_from_plan(plan: ResolvedPromptPlan) -> tuple[CreativeShotPayload, ...]:
    """Project DSPy plan shots into backend-neutral creative payloads."""
    if not isinstance(plan, ResolvedPromptPlan):
        raise TypeError("plan must be a ResolvedPromptPlan")
    result: list[CreativeShotPayload] = []
    seen: set[int] = set()
    for shot in plan.shots:
        number = int(shot.shot_number)
        if number in seen:
            raise ValueError(f"duplicate planned shot number: {number}")
        seen.add(number)
        result.append(CreativeShotPayload(
            shot_id=f"shot-{number:04d}",
            visible_action=_strip_compiler_syntax(shot.visible_action or shot.description),
            performance=_strip_compiler_syntax(shot.performance or plan.creative_intent),
            camera_behavior=_strip_compiler_syntax(shot.camera_behavior),
            environmental_motion=_strip_compiler_syntax(shot.environmental_motion),
            transition_intent=_strip_compiler_syntax(shot.transition_intent),
        ))
    return validate_creative_shots_against_plan(plan, result)


def _strip_compiler_syntax(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"<(?:Picture|Video|Audio)\s+\d+>", "", str(value), flags=re.IGNORECASE)
    text = re.sub(r"\b\d{2}:\d{2}(?:[:.]\d{2,3})\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


def validate_creative_shots_against_plan(
    plan: ResolvedPromptPlan,
    shots: Sequence[CreativeShotPayload],
) -> tuple[CreativeShotPayload, ...]:
    """Validate and order creative payloads against their enclosing plan.

    Shot IDs are derived from the plan's stable shot numbers.  Keeping this
    check at the plan boundary prevents a structurally valid payload from
    smuggling an unrelated shot into deterministic prompt compilation.
    """
    if not isinstance(plan, ResolvedPromptPlan):
        raise TypeError("plan must be a ResolvedPromptPlan")
    expected: list[str] = []
    for planned in plan.shots:
        shot_id = f"shot-{int(planned.shot_number):04d}"
        if shot_id in expected:
            raise ValueError(f"duplicate planned shot ID: {shot_id}")
        expected.append(shot_id)

    by_id: dict[str, CreativeShotPayload] = {}
    for shot in shots:
        if not isinstance(shot, CreativeShotPayload):
            raise TypeError("shots must contain CreativeShotPayload values")
        if shot.shot_id not in expected:
            raise ValueError(f"unknown shot ID: {shot.shot_id}")
        if shot.shot_id in by_id:
            raise ValueError(f"duplicate creative shot ID: {shot.shot_id}")
        by_id[shot.shot_id] = shot

    for shot_id in expected:
        if shot_id not in by_id:
            raise ValueError(f"missing creative shot payload: {shot_id}")
    return tuple(by_id[shot_id] for shot_id in expected)


def _time(value: Any) -> str:
    seconds = float(value)
    minutes, remainder = divmod(seconds, 60.0)
    return f"{int(minutes):02d}:{remainder:06.3f}"
