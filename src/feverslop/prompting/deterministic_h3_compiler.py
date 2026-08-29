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
H3_COMPILER_VERSION = 26


def plan_with_authoritative_relay(
    plan: ResolvedPromptPlan,
    relay_segments: Sequence[Mapping[str, Any]],
    *,
    language: str = "English",
    speaker_bindings: Sequence[Mapping[str, Any]] = (),
) -> ResolvedPromptPlan:
    """Make the persisted and judged plan agree with compiler-owned relay events."""
    if not relay_segments:
        return plan
    speaker_ids = _validated_speaker_ids(speaker_bindings)
    shots = []
    fields = (
        "description", "visible_action", "performance", "camera_behavior",
        "environmental_motion", "transition_intent",
    )
    for index, shot in enumerate(plan.shots):
        if index >= len(relay_segments):
            shots.append(shot)
            continue
        relay = relay_segments[index]
        content = _relay_vocal_content(relay)
        values: dict[str, str | None] = {}
        for field in fields:
            value = getattr(shot, field)
            if value is None:
                values[field] = None
                continue
            cleaned = _strip_authored_dialogue_markup(value)
            if content:
                cleaned = re.sub(re.escape(content), "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"(['\"])\s*\1", "", cleaned)
            cleaned = re.sub(r"\[[^]\r\n]+\]", "", cleaned)
            cleaned = _remove_authored_vocal_claims(f"[Shot 1] {cleaned}", relay)
            cleaned = re.sub(r"^\[Shot 1\]\s*", "", cleaned).strip()
            cleaned = " ".join(
                part for part in re.split(r"(?<=[.!?])\s+", cleaned)
                if not re.search(r"(?i)</?<?d\b|<</d>", part)
            ).strip()
            values[field] = cleaned or None
        if content:
            subject = _relay_speaker_label(relay, bound_subject_labels=set(speaker_ids))
            speaker_id = str(relay.get("speaker_id") or speaker_ids.get(subject) or "").strip()
            source = f"{subject} ({speaker_id})" if subject and speaker_id else "The audible voice"
            state = str(relay.get("state") or "").strip().casefold()
            verb = "sings" if state in {"singing", "vocals", "vocal"} else "says"
            event = f"{source} {verb}, <d>[{language or 'English'}] {_ensure_dialogue_punctuation(content)}</d>"
            values["description"] = " ".join(
                part for part in (values.get("description"), event) if part
            )
        if not values.get("description"):
            values["description"] = values.get("visible_action") or "The scene continues visually."
        shots.append(shot.model_copy(update=values))
    return plan.model_copy(update={"shots": shots})


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
        relay_segments: Sequence[Mapping[str, Any]] | None = None,
        speaker_bindings: Sequence[Mapping[str, Any]] | None = None,
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
                relay_segments=relay_segments,
                speaker_bindings=speaker_bindings,
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
        relay_segments: Sequence[Mapping[str, Any]] | None,
        speaker_bindings: Sequence[Mapping[str, Any]] | None,
    ) -> str:
        """Serialize LLM-authored fields into the MiniMax guide grammar."""
        if mode is PromptMode.R2V:
            bindings_by_audio = {
                str(binding.get("audio_label") or "").strip(): binding
                for binding in speaker_bindings or ()
                if str(binding.get("audio_label") or "").strip()
            }
            speaker_ids = _validated_speaker_ids(speaker_bindings or ())
            for relay in relay_segments or ():
                subject_label = str(relay.get("subject_label") or "").strip()
                speaker_id = str(relay.get("speaker_id") or "").strip()
                if subject_label and speaker_id:
                    existing = speaker_ids.get(subject_label)
                    if existing and existing != speaker_id:
                        raise ValueError(
                            f"conflicting speaker ID for {subject_label}: "
                            f"{existing} != {speaker_id}",
                        )
                    conflicting_subject = next((
                        label for label, value in speaker_ids.items()
                        if value == speaker_id and label != subject_label
                    ), "")
                    if conflicting_subject:
                        raise ValueError(
                            f"speaker ID {speaker_id} is already bound to {conflicting_subject}",
                        )
                    speaker_ids[subject_label] = speaker_id
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
                    copy_mode = _effective_audio_copy_mode(metadata)
                    name = str(metadata.get("name") or "audio reference").strip()
                    role = (
                        "original full-mix song used only as a rhythm and timing reference"
                        if name.casefold() == "full_mix" and copy_mode == "reference"
                        else {
                            "full_mix": "complete soundtrack",
                            "vocals": "vocal stem",
                        }.get(name.casefold(), name.replace("_", " "))
                    )
                    relationship = {
                        "fully_copy": "is fully copied as the target video's audio signal",
                        "partially_copy": "is partially copied into the target video's audio signal",
                    }.get(
                        copy_mode,
                        "is referenced for rhythm and timing without copying the source signal",
                    )
                    binding = bindings_by_audio.get(usage.reference_label)
                    if binding:
                        subject_label = str(binding.get("subject_label") or "").strip()
                        speaker_id = str(binding.get("speaker_id") or "").strip()
                        subject_lines.append(
                            f"{usage.reference_label} is the voice-timbre reference for "
                            f"{subject_label} ({speaker_id}); it is the {role} and {relationship}."
                        )
                    else:
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
                    _effective_audio_copy_mode(reference)
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
                    _effective_audio_copy_mode(metadata)
                    if str(metadata.get("kind") or "").lower() == "audio"
                    else "fully_preserved"
                )
                retention_lines.append(f"{label}: {marker} - reference is applied in the target video.")
            retention_lines.extend(usage_retention_lines)
            detailed_parts = [_remove_authored_dialogue_blocks(plan.style_opening or "")]
            rendered_shots = [
                _render_shot_with_references(
                    index,
                    shot,
                    plan,
                    reference_metadata=metadata_by_label,
                    final_shot=index == len(plan.shots),
                )
                for index, shot in enumerate(plan.shots, start=1)
            ]
            if relay_segments:
                detailed_parts[0] = _strip_authored_dialogue_markup(detailed_parts[0])
                rendered_shots = [
                    re.sub(
                        r"(?i)(<Subject\s+\d+>)\s*\(S\d+\)",
                        r"\1",
                        _strip_authored_dialogue_markup(shot_text),
                    )
                    for shot_text in rendered_shots
                ]
            for index, relay in enumerate(relay_segments or ()):
                if index >= len(rendered_shots):
                    break
                content = _relay_vocal_content(relay)
                if content:
                    rendered_shots[index] = re.sub(
                        re.escape(content),
                        "",
                        rendered_shots[index],
                        flags=re.IGNORECASE,
                    )
                    rendered_shots[index] = re.sub(
                        r"(?<=\s)[.!?](?=\s|$)",
                        "",
                        rendered_shots[index],
                    )
                rendered_shots[index] = _remove_authored_vocal_claims(
                    rendered_shots[index], relay,
                )
                rendered_shots[index] = _insert_authoritative_vocal_event(
                    rendered_shots[index], relay,
                    bound_speaker_ids=speaker_ids,
                )
            detailed_parts.extend(rendered_shots)
            vocal_audio_label = next((
                label
                for label, metadata in metadata_by_label.items()
                if str(metadata.get("kind") or "").casefold() == "audio"
                and (
                    _effective_audio_copy_mode(metadata) == "partially_copy"
                    or re.search(
                        r"\b(?:vocal|voice|dialogue|speech)\b",
                        " ".join((
                            str(metadata.get("name") or ""),
                            str(metadata.get("description") or ""),
                        )),
                        re.IGNORECASE,
                    )
                )
            ), None)
            detailed = _normalize_r2v_dialogue(
                "\n".join(detailed_parts),
                dialogue_language or "English",
                plan,
                vocal_audio_label=vocal_audio_label,
                canonical_sources_only=bool(relay_segments),
                bound_speaker_ids=speaker_ids,
            )
            for label, metadata in sorted(metadata_by_label.items()):
                copy_mode = _effective_audio_copy_mode(metadata)
                if (
                    str(metadata.get("kind") or "").casefold() == "audio"
                    and copy_mode in {"fully_copy", "partially_copy"}
                    and _audio_layer_kind(metadata) == "vocal"
                    and label not in detailed
                ):
                    detailed += " " + _copied_audio_layer_sentence(
                        label, copy_mode, "vocal layer",
                    )
            soundscape = _replace_subject_names(
                _remove_authored_dialogue_blocks(plan.overall_soundscape), plan,
            )
            for lyric in _dialogue_contents(detailed):
                soundscape = _remove_sentence_containing(soundscape, lyric)
            soundscape = _remove_authored_vocal_claims(soundscape, {})
            soundscape = _remove_music_sentences(soundscape)
            for label, metadata in sorted(metadata_by_label.items()):
                copy_mode = _effective_audio_copy_mode(metadata)
                if (
                    str(metadata.get("kind") or "").casefold() == "audio"
                    and copy_mode in {"fully_copy", "partially_copy"}
                    and _audio_layer_kind(metadata) == "ambience"
                    and label not in soundscape
                ):
                    soundscape += " " + _copied_audio_layer_sentence(
                        label, copy_mode, "ambience and sound-effects layer",
                    )
            if not soundscape.strip():
                soundscape = "No additional diegetic ambience or physical sound effects are specified."
            sections = [
                "subject_definitions:\n" + "\n".join(subject_lines),
                "summary: " + _summary_prefix(plan, metadata_by_label) + " "
                + _render_summary(plan, metadata_by_label),
                "retention_analysis:\n" + "\n".join(retention_lines),
                "detailed_description: " + detailed,
                "overall_soundscape: " + soundscape,
                "non_diegetic_music: " + _render_non_diegetic_music(
                    plan, metadata_by_label,
                ),
            ]
        else:
            detailed = "\n".join(
                _render_base_shot(
                    index, shot, plan, mode, final_shot=index == len(plan.shots),
                )
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
    reference_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    final_shot: bool = False,
) -> str:
    labels = _shot_reference_labels(shot, plan)
    description = _render_authored_shot_fields(shot, plan)
    missing = [label for label in labels if label not in description]
    if missing:
        visible = " and ".join(missing)
        verb = "is" if len(missing) == 1 else "are"
        description = (
            f"{description.rstrip('.')}. {visible} {verb} also present in the shot."
        )
    subject_sources = {
        label for subject in plan.subjects for label in subject.source_references
    }
    metadata_by_label = reference_metadata or {}
    standalone_labels = [
        label for label in shot.reference_labels
        if label not in subject_sources
        and str(metadata_by_label.get(label, {}).get("kind") or "").casefold() != "audio"
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
    copy_mode = _effective_audio_copy_mode(metadata)
    if copy_mode == "fully_copy":
        return f"{label} is fully copied as the complete soundtrack and timing reference"
    if copy_mode == "partially_copy":
        return f"the selected signal from {label} partially copied into the target video's audio"
    return (
        f"{label} is referenced for the target video's rhythm and timing "
        "without copying the source signal"
    )


def _effective_audio_copy_mode(metadata: Mapping[str, Any]) -> str:
    """Canonicalize legacy global-song metadata for a scene-level target."""
    raw = str(metadata.get("copy_mode") or "reference").casefold()
    identity = " ".join((
        str(metadata.get("name") or ""),
        str(metadata.get("description") or ""),
    )).casefold()
    if "full_mix" in identity and re.search(r"\b(?:original song|beat|rhythm)\b", identity):
        return "reference"
    return raw if raw in {"fully_copy", "partially_copy", "reference", "weak_reference"} else "reference"


def _audio_layer_kind(metadata: Mapping[str, Any]) -> str:
    identity = " ".join((
        str(metadata.get("name") or ""),
        str(metadata.get("description") or ""),
    )).casefold()
    if re.search(r"\b(?:vocal|voice|dialogue|speech|narration)\b", identity):
        return "vocal"
    if re.search(r"\b(?:ambience|ambient|sound effect|sfx|foley|room tone)\b", identity):
        return "ambience"
    return "music"


def _copied_audio_layer_sentence(label: str, copy_mode: str, layer: str) -> str:
    if copy_mode == "fully_copy":
        return f"The {layer} from {label} is fully copied into the target video."
    return f"The selected {layer} from {label} is partially copied into the target video."


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
        _effective_audio_copy_mode(item)
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
    summary = _replace_subject_names(
        _remove_authored_dialogue_blocks(plan.creative_intent), plan,
    ).strip().rstrip(".") + "."
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
        copy_mode = _effective_audio_copy_mode(metadata_by_label.get(label, {}))
        relationship = {
            "fully_copy": f"{label} is fully copied as the complete target-video audio track.",
            "partially_copy": f"Selected signal from {label} is partially copied into the target video.",
        }.get(copy_mode, f"{label} is referenced without copying the source signal.")
        summary += " " + relationship
    missing_subjects = [
        subject.label for subject in plan.subjects if subject.label not in summary
    ]
    if missing_subjects:
        summary += " The target video includes " + _english_join(missing_subjects) + "."
    return summary


def _english_join(values: Sequence[str]) -> str:
    items = [str(value).strip() for value in values if str(value).strip()]
    if len(items) < 2:
        return "".join(items)
    if len(items) == 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


def _render_non_diegetic_music(
    plan: ResolvedPromptPlan,
    metadata_by_label: Mapping[str, Mapping[str, Any]],
) -> str:
    parts = [_remove_authored_dialogue_blocks(plan.non_diegetic_music or "")]
    for label, metadata in sorted(metadata_by_label.items()):
        if str(metadata.get("kind") or "").casefold() != "audio":
            continue
        if _audio_layer_kind(metadata) != "music":
            continue
        copy_mode = _effective_audio_copy_mode(metadata)
        if copy_mode in {"reference", "weak_reference"} and plan.music_intent.value == "none":
            continue
        relationship = {
            "fully_copy": (
                f"{label} is directly reused as the complete audience-only score."
            ),
            "partially_copy": (
                f"The selected audience-only music layer from {label} is partially copied "
                "into the target video."
            ),
        }.get(
            copy_mode,
            f"{label} is referenced for the audience-only score's rhythm, tempo, and "
            "dynamic timing without copying the source signal.",
        )
        if label not in " ".join(parts):
            parts.append(relationship)
    return " ".join(part for part in parts if part) or "N/A"


def _render_base_shot(
    index: int,
    shot: Any,
    plan: ResolvedPromptPlan,
    mode: PromptMode,
    *,
    final_shot: bool,
) -> str:
    description = _render_authored_shot_fields(shot, plan)
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


def _render_authored_shot_fields(shot: Any, plan: ResolvedPromptPlan) -> str:
    """Join every LLM-authored creative field without inventing scene content."""
    values = (
        ("description", shot.description),
        ("visible_action", shot.visible_action),
        ("performance", shot.performance),
        ("camera_behavior", _render_camera_behavior(shot.camera_behavior) if shot.camera_behavior else ""),
        ("environmental_motion", shot.environmental_motion),
        ("transition_intent", shot.transition_intent),
    )
    parts: list[str] = []
    seen: set[str] = set()
    for field, value in values:
        text = _replace_subject_names(str(value or "").strip(), plan)
        if field == "visible_action" and re.match(r"^[^.!?]{1,80}['’]s\s", text):
            text = f"The shot shows {text}"
        key = re.sub(r"\W+", " ", text, flags=re.UNICODE).strip().casefold()
        if not text or not key or key in seen:
            continue
        seen.add(key)
        parts.append(_with_terminal_punctuation(text))
    return " ".join(parts)


def _with_terminal_punctuation(value: str) -> str:
    text = str(value).strip()
    if not text or text.endswith((".", "!", "?", "</d>")):
        return text
    return text + "."


def _render_camera_behavior(value: str) -> str:
    raw = str(value).strip()
    malformed = re.match(r"(?i)^the camera\s+the\s+(.+?)[.]?$", raw)
    if malformed:
        return f"The camera maintains the {malformed.group(1).rstrip('.')}."
    if re.match(r"(?i)^the camera\b", raw):
        return _with_terminal_punctuation(raw)
    text = _lower_initial(raw.rstrip("."))
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
    return _with_terminal_punctuation(raw)


def _normalize_r2v_dialogue(
    text: str,
    language: str,
    plan: ResolvedPromptPlan,
    *,
    vocal_audio_label: str | None = None,
    canonical_sources_only: bool = False,
    bound_speaker_ids: Mapping[str, str] | None = None,
) -> str:
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

    if not canonical_sources_only:
        normalized = re.sub(
            r"(?i)(\b(?:sings?|singing|says?|saying)(?:\s+the\s+(?:words?|lyrics?))?\s*)"
            r"(['\"])([^'\"\n]+)\2",
            quoted,
            normalized,
        )
    normalized = re.sub(r"</d>\s*[.?!](?=\s|$)", "</d>", normalized, flags=re.IGNORECASE)
    # Speaker IDs are compiler-owned. Authored prose cannot assign them, and
    # prose order is never used to infer a speaker.
    if not canonical_sources_only:
        for subject in plan.subjects:
            normalized = re.sub(
                rf"({re.escape(subject.label)})\s*\(S\d+\)",
                r"\1",
                normalized,
                flags=re.IGNORECASE,
            )
    if vocal_audio_label:
        normalized = re.sub(
            r"</d>(?!\s+from\s+<Audio\s+\d+>)",
            f"</d> from {vocal_audio_label}.",
            normalized,
            flags=re.IGNORECASE,
        )
    return normalized


def _insert_authoritative_vocal_event(
    shot_text: str,
    relay: Mapping[str, Any],
    *,
    bound_speaker_ids: Mapping[str, str] | None = None,
) -> str:
    """Insert exact relay lyrics as compiler-owned dialogue syntax."""
    state = str(relay.get("state") or "").strip().casefold()
    if state not in {"singing", "dialogue", "speech", "spoken", "vocals", "vocal"}:
        return shot_text
    content = _relay_vocal_content(relay)
    if not content:
        return shot_text

    # Repair LLM-authored or judge-authored dialogue markup around the known
    # source words. The compiler owns the tags and language label.
    normalized = re.sub(r"</?d(?:\s+[^>]*)?>", "", shot_text, flags=re.IGNORECASE)
    tagged = f"<d>{content}</d>"
    content_pattern = re.compile(re.escape(content), re.IGNORECASE)
    normalized = content_pattern.sub("", normalized)
    normalized = re.sub(
        r"\b(?:performing\s+the\s+lyrics|lyrics|dialogue)\s*:\s*(?=[,.;]|$)",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\s+([,.;])", r"\1", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    speaker = _relay_speaker_label(
        relay, bound_subject_labels=set(bound_speaker_ids or {}),
    )
    speaker_id = str(relay.get("speaker_id") or "").strip()
    if speaker and not speaker_id:
        speaker_id = str((bound_speaker_ids or {}).get(speaker) or "").strip()
    verb = "sings" if state in {"singing", "vocals", "vocal"} else "says"
    source = f"{speaker} ({speaker_id})" if speaker and speaker_id else speaker
    source = source or "The audible voice"
    return f"{normalized.rstrip()} {source} {verb}, {tagged}"


def _remove_authored_vocal_claims(
    shot_text: str,
    relay: Mapping[str, Any],
) -> str:
    """Remove creative vocal claims when the relay owns the audible event."""
    marker_match = re.match(
        r"^(\[Shot\s+\d+\](?:\s+At\s+[^,]+,)?)\s*(.*)$",
        shot_text.strip(),
        flags=re.IGNORECASE | re.DOTALL,
    )
    marker = marker_match.group(1) if marker_match else ""
    body = marker_match.group(2) if marker_match else shot_text.strip()
    parts = re.split(r"(?<=[.!?])\s+", body)
    kept = [cleaned for part in parts if (cleaned := _remove_vocal_clause(part))]
    retained = " ".join(kept).strip()
    return " ".join(part for part in (marker, retained) if part).strip()


_VOCAL_VERB = re.compile(
    r"(?i)\b(?:sings?|singing|says?|speaks?|speaking|performs?|performing|"
    r"whispers?|whispering|chants?|chanting|shouts?|shouting|mouths|"
    r"lip[- ]?syncs?|lips?\s+(?:move|moving)|delivers?)\b",
)


def _remove_vocal_clause(sentence: str) -> str:
    """Remove authored vocal clauses while retaining independent visual clauses."""
    retained = sentence.strip()
    for _ in range(10):
        verb = _VOCAL_VERB.search(retained)
        if not verb:
            break
        before = retained[:verb.start()].rstrip()
        if verb.group().casefold() in {"say", "says", "said"} and re.search(
            r"(?i)\b(?:display|panel|screen|sign|caption|title|text)\b",
            before,
        ):
            break
        split = list(re.finditer(r"(?i)\b(?:and|while|as)\s+", before))
        boundary = split[-1] if split else None
        prefix = before[:boundary.start()].rstrip(" ,;") if boundary else ""
        actor = before[boundary.end():].strip() if boundary else before.strip()
        suffix_match = re.search(
            r"(?i)\b(and|while|as)\s+(.+)$",
            retained[verb.end():],
        )
        suffix = ""
        if suffix_match:
            conjunction = suffix_match.group(1).casefold()
            suffix = suffix_match.group(2).strip()
            if conjunction == "and" and actor:
                suffix = f"{actor} {suffix}"
        retained = " ".join(part for part in (prefix, suffix) if part).strip()
    retained = re.sub(r"\s+([,.;!?])", r"\1", retained)
    if retained and sentence.rstrip().endswith((".", "!", "?")) and not retained.endswith((".", "!", "?")):
        retained += sentence.rstrip()[-1]
    return retained[:1].upper() + retained[1:] if retained else ""


def _strip_authored_dialogue_markup(text: str) -> str:
    """Keep authored prose while removing compiler-owned dialogue markup."""
    normalized = re.sub(
        r"<d\s*>\s*(?:\[[^]\r\n]+\]\s*)?",
        "",
        str(text),
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"</d\s*>", "", normalized, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", normalized).strip()


def _remove_authored_dialogue_blocks(text: str) -> str:
    """Remove LLM-owned dialogue payloads from sections where dialogue is forbidden."""
    normalized = re.sub(
        r"<d\s*>.*?</d\s*>",
        "",
        str(text),
        flags=re.IGNORECASE | re.DOTALL,
    )
    return re.sub(r"\s+", " ", normalized).strip()


def _relay_vocal_content(relay: Mapping[str, Any]) -> str:
    for key in ("lyrics", "dialogue", "text"):
        value = str(relay.get(key) or "").strip()
        if value:
            return value
    prompt = str(relay.get("prompt") or "").strip()
    match = re.search(
        r"\b(?:performing\s+the\s+lyrics|lyrics|dialogue|says?)\s*:\s*(.+?)\s*$",
        prompt,
        re.IGNORECASE,
    )
    if not match:
        return ""
    value = match.group(1).strip().strip("'\"")
    # Section labels describe song structure and are not spoken words.
    return "" if re.fullmatch(r"\[[^]]+\]", value) else value


def _relay_speaker_label(
    relay: Mapping[str, Any],
    *,
    bound_subject_labels: set[str],
) -> str:
    subject_label = str(relay.get("subject_label") or "").strip()
    return subject_label if subject_label in bound_subject_labels else ""


def _validated_speaker_ids(
    bindings: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    by_subject: dict[str, str] = {}
    by_speaker: dict[str, str] = {}
    for binding in bindings:
        subject_label = str(binding.get("subject_label") or "").strip()
        speaker_id = str(binding.get("speaker_id") or "").strip()
        if not subject_label or not speaker_id:
            continue
        existing_id = by_subject.get(subject_label)
        if existing_id and existing_id != speaker_id:
            raise ValueError(
                f"subject {subject_label} is bound to both {existing_id} and {speaker_id}"
            )
        existing_subject = by_speaker.get(speaker_id)
        if existing_subject and existing_subject != subject_label:
            raise ValueError(
                f"speaker ID {speaker_id} is bound to both "
                f"{existing_subject} and {subject_label}"
            )
        by_subject[subject_label] = speaker_id
        by_speaker[speaker_id] = subject_label
    return by_subject


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
    text = re.sub(
        r"(?:,\s*)?(?:while\s+)?[^,.;]*\bvocals?\b[^.;]*[.;]?",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    parts = re.split(r"(?<=[.!?;])\s+", text.strip())
    music = re.compile(r"\b(?:musical track|full mix|song|background music)\b", re.IGNORECASE)
    retained = " ".join(part for part in parts if not music.search(part)).strip(" ;")
    return retained[:1].upper() + retained[1:] if retained else ""


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
