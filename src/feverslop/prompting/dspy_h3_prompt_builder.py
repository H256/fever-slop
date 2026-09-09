from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from feverslop.domain.performance_sync import select_performance_audio_paths
from feverslop.domain.h3_audio_delivery import H3AudioDelivery
from feverslop.domain.h3_prompt_checkpoint import H3PromptCheckpointInput
from feverslop.domain.locked_scene_facts import LockedSceneFacts, locked_scene_facts_from_scene
from feverslop.prompting.dspy_h3_models import (
    AudioSubjectBinding,
    H3PromptSections,
    MusicIntent,
)
from feverslop.prompting.deterministic_h3_compiler import (
    H3_COMPILER_NAME,
    H3_COMPILER_VERSION,
    plan_with_authoritative_relay,
    DeterministicH3Compiler,
    creative_shots_from_plan,
)
from feverslop.prompting.guide_loader import load_markdown_guide
from feverslop.prompting.planning_payload import compact_planning_payload
from feverslop.prompting.prompt_contract_validation import (
    PromptContractError,
    PromptContractIssue,
    validate_prompt_contract,
    validate_h3_prompt_contract,
    validate_h3_prompt_shape,
    validate_performance_phases,
)
from feverslop.prompting.subject_directive_planning import (
    project_directives_to_prompt,
    subject_directives_from_scene,
)

if TYPE_CHECKING:
    from feverslop.ports.h3_prompt_checkpoints import H3PromptCheckpointPort


def _reference(
    *,
    label: str,
    source: str | Path,
    kind: str,
    name: str,
    description: str | None = None,
    role: str = "general",
) -> dict[str, str]:
    source_text = str(source).replace("\\", "/")
    return {
        "label": label,
        "source": source_text,
        "kind": kind,
        "name": name,
        "description": description if description is not None else f"Preserved {kind} reference from {Path(source_text).name}.",
        "role": role,
    }


def _reference_source_key(source: str | Path, reference_root: Path | None) -> str:
    """Return a stable key so relative and absolute project paths deduplicate."""
    path = Path(str(source))
    if reference_root is not None and not path.is_absolute():
        path = reference_root / path
    try:
        return path.resolve(strict=False).as_posix().casefold()
    except OSError:
        return str(path).replace("\\", "/").casefold()


def _audio_subject_bindings(
    references: dict[str, Any],
    *,
    available_stems: set[str] | None = None,
) -> dict[str, dict[str, str]]:
    """Validate explicit stem-to-subject bindings without inventing subjects."""
    raw = references.get("audio_subject_bindings") or {}
    if isinstance(raw, list):
        raw = {str(item.get("stem") or ""): item for item in raw if isinstance(item, dict)}
    if not isinstance(raw, dict):
        raise ValueError("audio_subject_bindings must be an object or list")
    actor_ids = [str(value) for value in references.get("actor_ids") or []]
    actor_labels = {value: f"<Subject {index}>" for index, value in enumerate(actor_ids, start=1)}
    result: dict[str, dict[str, str]] = {}
    seen_subjects: set[str] = set()
    seen_speaker_ids: dict[str, str] = {}
    for stem, value in raw.items():
        stem_name = str(stem).strip()
        if not stem_name or not isinstance(value, dict):
            raise ValueError("each audio subject binding requires a stem and object value")
        if stem_name == "full_mix":
            raise ValueError("full_mix is global audio and cannot bind to a subject")
        if available_stems is not None and stem_name not in available_stems:
            raise ValueError(f"audio binding refers to an unselected stem: {stem_name}")
        subject_id = str(value.get("subject_id") or value.get("subject") or "").strip()
        subject_label = str(value.get("subject_label") or actor_labels.get(subject_id) or "").strip()
        if not subject_label:
            raise ValueError(f"audio binding for {stem_name!r} has no known subject")
        if subject_label not in actor_labels.values():
            raise ValueError(f"audio binding refers to an unknown subject: {subject_label}")
        if subject_label in seen_subjects:
            raise ValueError(f"multiple audio stems bind to subject {subject_label}")
        seen_subjects.add(subject_label)
        speaker_id = str(value.get("speaker_id") or "").strip()
        if speaker_id and not re.fullmatch(r"S[1-9][0-9]*", speaker_id):
            raise ValueError(f"invalid speaker_id for {stem_name!r}: {speaker_id}")
        if speaker_id and int(speaker_id[1:]) > len(actor_ids):
            raise ValueError(f"speaker_id does not identify a visible subject: {speaker_id}")
        if stem_name == "vocals" and not speaker_id:
            raise ValueError("vocal audio binding requires speaker_id")
        other_subject = seen_speaker_ids.get(speaker_id) if speaker_id else None
        if other_subject and other_subject != subject_label:
            raise ValueError(
                f"speaker ID {speaker_id} is bound to both "
                f"{other_subject} and {subject_label}"
            )
        if speaker_id:
            seen_speaker_ids[speaker_id] = subject_label
        result[stem_name] = {
            "subject_label": subject_label,
            "speaker_id": speaker_id,
            "subject_id": subject_id,
        }
    return result


def _speaker_bindings_for_compile(
    *,
    segment: dict[str, Any],
    references: list[dict[str, Any]],
    stored_bindings: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    """Return explicit generic speaker bindings, rebuilding stale checkpoint data."""
    audio_references = {
        str(reference.get("name") or "").strip(): str(reference.get("label") or "").strip()
        for reference in references
        if str(reference.get("kind") or "").casefold() == "audio"
        and str(reference.get("name") or "").strip()
        and str(reference.get("label") or "").strip()
    }
    raw = _audio_subject_bindings(
        segment.get("references") or {},
        available_stems=set(audio_references),
    )
    current = [
        {
            "audio_label": audio_references[stem],
            "stem": stem,
            "subject_label": binding["subject_label"],
            "speaker_id": binding["speaker_id"],
        }
        for stem, binding in raw.items()
        if stem in audio_references and binding.get("speaker_id")
    ]
    relay = segment.get("performance_intervals") or (segment.get("ltx") or {}).get("prompt_relay") or segment.get("prompt_relay") or []
    relay = [event for phase in relay for event in phase.get("vocal_events") or [phase]]
    labels = {actor: f"<Subject {i}>" for i, actor in enumerate((segment.get("references") or {}).get("actor_ids") or [], 1)}
    for item in relay:
        if not isinstance(item, dict):
            continue
        if item.get("offscreen"):
            continue
        subject_label = str(item.get("subject_label") or labels.get(item.get("subject_id")) or "").strip()
        speaker_id = str(item.get("speaker_id") or "").strip()
        if subject_label and speaker_id and not any(
            binding["subject_label"] == subject_label
            and binding["speaker_id"] == speaker_id
            for binding in current
        ):
            current.append({
                "subject_label": subject_label,
                "speaker_id": speaker_id,
            })
    _validate_speaker_binding_bijection(current, source="current")
    if not stored_bindings:
        return sorted(current, key=_speaker_binding_sort_key)
    stored = [
        {
            key: str(binding.get(key) or "").strip()
            for key in ("audio_label", "stem", "subject_label", "speaker_id")
            if str(binding.get(key) or "").strip()
        }
        for binding in stored_bindings
        if isinstance(binding, dict)
    ]
    actor_count = len((segment.get("references") or {}).get("actor_ids") or [])
    valid_audio_labels = set(audio_references.values())
    for binding in stored:
        subject_label = binding.get("subject_label", "")
        speaker_id = binding.get("speaker_id", "")
        subject_match = re.fullmatch(r"<Subject\s+([1-9][0-9]*)>", subject_label)
        if not subject_match or (actor_count and int(subject_match.group(1)) > actor_count):
            raise ValueError(f"stored speaker binding has unknown subject: {subject_label}")
        if not re.fullmatch(r"S[1-9][0-9]*", speaker_id):
            raise ValueError(f"stored speaker binding has invalid speaker ID: {speaker_id}")
        audio_label = binding.get("audio_label", "")
        if audio_label and audio_label not in valid_audio_labels:
            raise ValueError(f"stored speaker binding has unknown audio reference: {audio_label}")
    _validate_speaker_binding_bijection(stored, source="stored")
    current = sorted(current, key=_speaker_binding_sort_key)
    stored = sorted(stored, key=_speaker_binding_sort_key)
    if stored != current:
        raise ValueError("stored speaker binding conflicts with current config")
    return stored


def _speaker_binding_sort_key(binding: dict[str, str]) -> tuple[str, str, str, str]:
    return tuple(
        str(binding.get(key) or "")
        for key in ("subject_label", "speaker_id", "audio_label", "stem")
    )


def _validate_speaker_binding_bijection(
    bindings: list[dict[str, str]],
    *,
    source: str,
) -> None:
    by_subject: dict[str, str] = {}
    by_speaker: dict[str, str] = {}
    for binding in bindings:
        subject_label = str(binding.get("subject_label") or "").strip()
        speaker_id = str(binding.get("speaker_id") or "").strip()
        if not subject_label or not speaker_id:
            continue
        other_id = by_subject.get(subject_label)
        if other_id and other_id != speaker_id:
            raise ValueError(
                f"{source} subject {subject_label} has multiple speaker IDs"
            )
        other_subject = by_speaker.get(speaker_id)
        if other_subject and other_subject != subject_label:
            raise ValueError(
                f"{source} speaker ID {speaker_id} is bound to multiple subjects"
            )
        by_subject[subject_label] = speaker_id
        by_speaker[speaker_id] = subject_label


def _audio_description(name: str, binding: dict[str, str] | None) -> str:
    if name == "full_mix":
        return "full_mix - original song for beat and rhythm continuity"
    if binding is None:
        return f"{name} stem; no subject binding was supplied"
    speaker = f" ({binding['speaker_id']})" if binding.get("speaker_id") else ""
    return f"{name} stem bound to {binding['subject_label']}{speaker}"


def _scene_audio_copy_mode(name: str, description: str) -> str:
    """Classify scene-level audio without claiming the whole song is copied 1:1."""
    identity = f"{name} {description}".casefold()
    return "reference" if "full_mix" in identity else "partially_copy"


def _is_full_mix_audio(name: str, description: str) -> bool:
    return "full_mix" in f"{name} {description}".casefold()


def _normalize_resolved_scene_references(
    references: list[dict[str, Any]],
    *,
    audio_delivery: H3AudioDelivery | None = None,
) -> list[dict[str, Any]]:
    """Upgrade stale scene-level full-song metadata before compile and judge."""
    normalized: list[dict[str, Any]] = []
    for reference in references:
        item = dict(reference)
        if item.get("kind") == "audio" and _is_full_mix_audio(
            str(item.get("name") or ""),
            str(item.get("description") or ""),
        ):
            item["copy_mode"] = (
                "fully_copy"
                if audio_delivery and audio_delivery.copies_to_output
                else "reference"
            )
        normalized.append(item)
    return normalized


def _normalize_plan_audio_usage(
    plan: Any,
    references: list[dict[str, Any]],
    *,
    audio_delivery: H3AudioDelivery | None = None,
) -> Any:
    """Align planner-owned audio wording with deterministic scene metadata."""
    by_label = {str(item.get("label")): item for item in references}
    usages = []
    for usage in plan.reference_usage:
        reference = by_label.get(usage.reference_label, {})
        if (
            str(reference.get("kind") or "").casefold() == "audio"
            and str(reference.get("copy_mode") or "").casefold() == "reference"
            and not (audio_delivery and audio_delivery.conditions_generation)
        ):
            usage = usage.model_copy(update={
                "purpose": "audio reference",
                "details": (
                    "Use the supplied original song only for beat, rhythm, and dynamic "
                    "timing continuity without copying the source signal."
                ),
            })
        usages.append(usage)
    updates: dict[str, Any] = {"reference_usage": usages}
    if audio_delivery and audio_delivery.conditions_generation and any(
        str(reference.get("kind") or "").casefold() == "audio"
        for reference in references
    ):
        updates["music_intent"] = MusicIntent.NONE
    return plan.model_copy(update=updates)


def _scene_references(
    segment: dict[str, Any],
    audio_paths: dict[str, Path] | None,
    reference_root: Path | None,
    mode: str = "r2v",
    audio_delivery: H3AudioDelivery | None = None,
) -> tuple[list[dict[str, str]], list[Path]]:
    references = segment.get("references") or {}
    selected_audio_paths = (
        select_performance_audio_paths(segment, audio_paths, max_stems=2)
        if audio_paths
        else {}
    )
    relay = segment.get("performance_intervals") or (segment.get("ltx") or {}).get("prompt_relay") or segment.get("prompt_relay") or []
    fully_instrumental = bool(relay) and all(
        str(item.get("state") or "").strip().lower() == "instrumental"
        for item in relay
    )
    audio_tags = references.get("_stem_audio_tags") or {}
    audio_bindings = _audio_subject_bindings(references)
    audio_tags_by_key = {
        _reference_source_key(source, reference_root): str(description)
        for source, description in audio_tags.items()
    }
    result: list[dict[str, str]] = []
    images: list[Path] = []
    seen: dict[str, set[str]] = {"picture": set(), "video": set(), "audio": set()}

    def add_reference(reference: dict[str, str], image_path: Path | None = None) -> None:
        source = reference["source"]
        kind = reference["kind"]
        source_key = _reference_source_key(source, reference_root)
        if source_key in seen[kind]:
            return
        seen[kind].add(source_key)

        # Labels must be derived from the references that actually survive
        # deduplication. The DSPy generator resolves labels in this same per-kind
        # order, so canonicalizing here guarantees that the returned backend
        # reference slots and the labels embedded in the generated prompt agree.
        canonical_reference = dict(reference)
        kind_number = 1 + sum(item["kind"] == kind for item in result)
        canonical_reference["label"] = f"<{kind.title()} {kind_number}>"
        result.append(canonical_reference)
        if image_path is not None and image_path.is_file():
            images.append(image_path)

    actor_selection_present = "actor_ids" in references
    actor_ids = references.get("actor_ids") or []
    actor_paths = (
        references.get("actor_sheet_paths") or references.get("actor_msr_paths") or []
        if not actor_selection_present or actor_ids
        else []
    )
    actor_metadata = {
        str(item.get("id") or item.get("name") or ""): item
        for item in references.get("actor_reference_descriptions") or []
        if isinstance(item, dict)
    }
    for index, source in enumerate(actor_paths, start=1):
        actor_id = str(actor_ids[index - 1]) if index <= len(actor_ids) else f"Actor {index}"
        metadata = actor_metadata.get(actor_id) or {}
        name = str(metadata.get("name") or actor_id)
        path = Path(source)
        image_path = path if path.is_absolute() or reference_root is None else reference_root / path
        actor_reference = _reference(
            label=f"<Picture {index}>",
            source=path,
            kind="picture",
            name=name,
            description=(
                str(metadata.get("visual_description") or metadata.get("image_prompt") or "").strip()
                if metadata
                else ("" if image_path.is_file() else None)
            ),
            role="subject",
        )
        actor_reference["id"] = actor_id
        add_reference(actor_reference, image_path)

    location = references.get("location_sheet_path") or references.get("location_msr_path")
    if location:
        path = Path(location)
        image_path = path if path.is_absolute() or reference_root is None else reference_root / path
        location_description = references.get("location_reference_description") or {}
        add_reference(_reference(
            label=f"<Picture {len(result) + 1}>",
            source=path,
            kind="picture",
            name=str(
                location_description.get("name")
                or references.get("location_id")
                or "Location",
            ),
            description=(
                str(
                    location_description.get("visual_description")
                    or location_description.get("image_prompt")
                    or "",
                ).strip()
                or ("" if image_path.is_file() else None)
            ),
            role="environment",
        ), image_path)

    for index, path_value in enumerate(references.get("reference_image_paths") or []):
        path = Path(path_value)
        image_path = path if path.is_absolute() or reference_root is None else reference_root / path
        role = "subject"
        if (mode == "i2v" and index == 0) or (mode == "fl2v" and index == 0):
            role = "first_frame"
        elif (mode == "fl2v" and index == 1) or (mode == "l2v" and index == 0):
            role = "last_frame"
        add_reference(_reference(
            label="<Picture pending>",
            source=path,
            kind="picture",
            name=path.stem,
            description="" if image_path.is_file() else None,
            role=role,
        ), image_path)

    for path_value in references.get("reference_video_paths") or []:
        path = Path(path_value)
        add_reference(_reference(
            label=f"<Video {len([ref for ref in result if ref['kind'] == 'video']) + 1}>",
            source=path,
            kind="video",
            name=path.stem,
            role="motion",
        ))

    pending_audio_references: list[dict[str, str]] = []
    for path_value in references.get("reference_audio_paths") or []:
        path = Path(path_value)
        tag = audio_tags_by_key.get(_reference_source_key(path, reference_root), "")
        if selected_audio_paths and tag:
            # Re-add managed stems below in the exact role-specific order so
            # H3 labels match the render-plan/backend slot order.
            continue
        if fully_instrumental and "vocal" in tag.casefold() and "full_mix" not in tag.casefold():
            continue
        description = tag or _audio_description(path.stem, audio_bindings.get(path.stem))
        copy_mode = _scene_audio_copy_mode(path.stem, description)
        if (
            audio_delivery is not None
            and audio_delivery.copies_to_output
            and _is_full_mix_audio(path.stem, description)
        ):
            copy_mode = "fully_copy"
        if tag and path.stem in audio_bindings:
            binding = audio_bindings[path.stem]
            speaker = f" ({binding['speaker_id']})" if binding.get("speaker_id") else ""
            description = f"{tag}; bound to {binding['subject_label']}{speaker}"
        pending_audio_references.append(_reference(
            label=f"<Audio {len([ref for ref in result if ref['kind'] == 'audio']) + 1}>",
            source=path,
            kind="audio",
            name=path.stem,
            description=description,
            role="audio_reuse",
        ) | {"copy_mode": copy_mode})

    for index, (name, source) in enumerate(selected_audio_paths.items(), start=1):
        if fully_instrumental and name == "vocals":
            continue
        add_reference(_reference(
            label=f"<Audio {len([ref for ref in result if ref['kind'] == 'audio']) + 1}>",
            source=source,
            kind="audio",
            name=name,
            description=_audio_description(name, audio_bindings.get(name)),
            role="audio_reuse",
        ) | {
            "copy_mode": (
                "fully_copy"
                if audio_delivery is not None
                and audio_delivery.copies_to_output
                and _is_full_mix_audio(name, _audio_description(name, audio_bindings.get(name)))
                else _scene_audio_copy_mode(name, _audio_description(name, audio_bindings.get(name)))
            ),
        })

    for reference in pending_audio_references:
        add_reference(reference)

    available_stems = {str(reference["name"]) for reference in result if reference["kind"] == "audio"}
    _audio_subject_bindings(references, available_stems=available_stems)
    return result, images


def _safe_error_message(error: BaseException) -> str:
    """Keep multimodal provider errors useful without dumping image payloads."""
    message = str(error)
    message = re.sub(
        r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+",
        "<embedded image omitted>",
        message,
    )
    message = re.sub(
        r"[A-Za-z0-9+/]{256,}={0,2}",
        "<binary payload omitted>",
        message,
    )
    message = message.replace(
        "]]>&lt;CUSTOM-TYPE-END-IDENTIFIER&gt;",
        "]]><CUSTOM-TYPE-END-IDENTIFIER>",
    )
    return message[:1000]


def _normalize_relay_segments(segment: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert LTX frame relays into bounded, model-neutral timed shots."""
    relay = segment.get("performance_intervals") or (segment.get("ltx") or {}).get("prompt_relay") or segment.get("prompt_relay") or []
    if not relay:
        return []
    fps = float(segment.get("fps") or 24)
    duration_value = segment.get("duration_seconds") or segment.get("duration")
    duration = float(duration_value) if duration_value is not None else None
    shots = []
    for index, item in enumerate(relay, start=1):
        if item.get("performance_phase") and "start_seconds" in item and "end_seconds" in item:
            start, end = float(item["start_seconds"]), float(item["end_seconds"])
        elif item.get("performance_phase") and "start" in item and "end" in item:
            origin = float(segment.get("abs_start_seconds", segment.get("start")) or 0)
            start, end = float(item["start"]) - origin, float(item["end"]) - origin
        else:
            start = float(item["frame_start"]) / fps
            end = float(item["frame_end"]) / fps
        if duration is not None:
            start = min(start, duration)
            end = min(end, duration)
        if end <= start:
            continue
        shot = {
            "shot": index,
            "start_seconds": start,
            "end_seconds": end,
            "state": str(item.get("state") or "").strip(),
            "prompt": str(item.get("prompt") or "").strip(),
        }
        for key in ("performance_phase", "performance_intervals_version", "word_timestamps", "acoustically_verified", "reason_codes", "performance_conflicts"):
            if key in item:
                shot[key] = item[key]
        if item.get("performance_phase"):
            shot["word_time_origin"] = float(segment.get("abs_start_seconds", segment.get("start")) or 0)
        source_prompt = str(item.get("source_prompt") or "").strip()
        if source_prompt:
            shot["source_prompt"] = source_prompt
        for key in (
            "lyrics", "dialogue", "text",
            "subject_id", "subject_label", "speaker_id", "speaker_description",
        ):
            value = str(item.get(key) or "").strip()
            if value:
                shot[key] = value
        events = item.get("vocal_events") or []
        if events:
            labels = {actor: f"<Subject {i}>" for i, actor in enumerate((segment.get("references") or {}).get("actor_ids") or [], 1)}
            for event in events:
                resolved = dict(shot)
                for key in ("lyrics", "word_timestamps", "subject_id", "subject_label", "speaker_id", "offscreen"):
                    resolved.pop(key, None)
                    if key in event:
                        resolved[key] = event[key]
                if resolved.get("subject_id") in labels:
                    resolved.setdefault("subject_label", labels[resolved["subject_id"]])
                shots.append(resolved)
        else:
            if "offscreen" in item:
                shot["offscreen"] = item["offscreen"]
            shots.append(shot)
    return shots


# Relay states that represent a sung vocal event, i.e. the states the
# deterministic compiler turns into a "sings" event and that therefore belong
# to the "vocals" audio stem rather than a spoken/dialogue stem.
_SING_RELAY_STATES = frozenset({"singing", "vocals", "vocal"})


def _stamp_relay_speaker_binding(
    relay_segments: list[dict[str, Any]],
    raw_bindings: dict[str, dict[str, str]],
) -> None:
    """Bind sung relay windows to the vocal stem's on-screen subject.

    The deterministic compiler only anchors an event's source to a visible
    subject when the relay shot carries a ``subject_label`` and ``speaker_id``;
    otherwise it falls back to the unanchored ``The audible voice`` source, which
    leaves the model free to pick any mouth (or none) and breaks lip-sync. The
    relay producer never emits those keys, so even a correctly bound vocal scene
    compiled to an unanchored source. Stamping the already-validated binding here
    is the single point that fixes both the primary build and the structured
    resume path without inventing subjects.
    """
    vocal = raw_bindings.get("vocals") or {}
    subject_label = str(vocal.get("subject_label") or "").strip()
    speaker_id = str(vocal.get("speaker_id") or "").strip()
    if not subject_label or not speaker_id:
        return
    for shot in relay_segments:
        if shot.get("offscreen"):
            continue
        if str(shot.get("state") or "").strip().casefold() not in _SING_RELAY_STATES:
            continue
        if not str(shot.get("subject_label") or "").strip():
            shot["subject_label"] = subject_label
        if not str(shot.get("speaker_id") or "").strip():
            shot["speaker_id"] = speaker_id


def _relay_vocal_binding(
    relay_segments: list[dict[str, Any]],
    raw_bindings: dict[str, dict[str, str]],
) -> tuple[int, str | None]:
    """Return the sung relay-event count and bound vocal subject label."""
    relay_vocal_events = sum(
        1
        for shot in relay_segments
        if str(shot.get("state") or "").strip().casefold() in _SING_RELAY_STATES
        and (not shot.get("performance_phase") or str(shot.get("lyrics") or "").strip())
    )
    bound_vocal_subject = (
        str((raw_bindings.get("vocals") or {}).get("subject_label") or "").strip() or None
    )
    if any(shot.get("offscreen") for shot in relay_segments) or len({shot.get("subject_label") for shot in relay_segments if shot.get("subject_label")}) > 1:
        bound_vocal_subject = None
    return relay_vocal_events, bound_vocal_subject


def _format_relay_shots(shots: list[dict[str, Any]]) -> str:
    if not shots:
        return ""
    performance = any(shot.get("performance_phase") for shot in shots)
    lines = ["Performance timing within one continuous camera shot:" if performance else "Temporal shot directions:"]
    for shot in shots:
        state = f" ({shot['state']})" if shot.get("state") else ""
        lines.append(
            f"[{'Performance phase' if performance else 'Shot'} {shot['shot']}, {shot['start_seconds']:.2f}-{shot['end_seconds']:.2f}sec]"
            f"{state} {shot['prompt']}",
        )
        source_prompt = shot.get("source_prompt")
        if source_prompt and source_prompt != shot.get("prompt"):
            lines.append(f"Required action and props to preserve: {source_prompt}")
    return "\n".join(lines)


class DspyH3PromptBuilder:
    """Adapter around the DSPy scene generator used by the H3 R2V pipeline."""

    def __init__(
        self,
        generator: Callable[[dict[str, Any]], Any],
        *,
        reference_root: Path | None = None,
        # Compatibility callers may retain concept-only fallback; production
        # construction must pass False so DSPy failures are surfaced.
        allow_fallback: bool = True,
        reporter: Any = None,
    ):
        self.generator = generator
        self.reference_root = reference_root
        self.allow_fallback = allow_fallback
        self.reporter = reporter

    def set_reporter(self, reporter: Any) -> None:
        self.reporter = reporter

    def _resolve_audio_delivery(self, references, delivery, segment, reference_root=None):
        if not delivery.workflow_path:
            return references
        from feverslop.domain.h3_audio_delivery import resolve_h3_audio_sources, h3_audio_timing_window
        audio = []
        for reference in references:
            if reference.get("kind") == "audio":
                source = Path(reference["source"])
                if not source.is_absolute() and (reference_root or self.reference_root):
                    source = Path(reference_root or self.reference_root) / source
                audio.append({**reference, "source": str(source)})
        timing_scene = {**segment, "abs_start_seconds": segment.get("abs_start_seconds", segment.get("start", 0))}
        if "abs_end_seconds" not in timing_scene and "end" in segment:
            timing_scene["abs_end_seconds"] = segment["end"]
        duration = segment.get("duration_seconds", segment.get("duration"))
        if self.reporter is not None:
            self.reporter.message("Resolving H3 audio source roles before prompt compilation")
        sources = resolve_h3_audio_sources(delivery, audio, h3_audio_timing_window(timing_scene, duration))
        by_label = {source["label"]: source for source in sources}
        result = []
        for reference in references:
            item = dict(reference)
            if item.get("kind") == "audio":
                source = by_label[item["label"]]
                item["audio_delivery"] = source
                item["delivery_roles"] = source["roles"]
                item["copy_mode"] = (
                    "fully_copy" if source["name"] == "full_mix" else "partially_copy"
                ) if "output_copy" in source["roles"] else "reference"
            result.append(item)
        if self.reporter is not None:
            self.reporter.message(f"Validated H3 audio roles for {len(sources)} sources")
        return result

    def checkpoint_revision(self) -> dict[str, Any]:
        revision: dict[str, Any] = {
            "contract": 3,
            "compiler": H3_COMPILER_NAME,
            "compiler_version": H3_COMPILER_VERSION,
            "generator": f"{type(self.generator).__module__}.{type(self.generator).__qualname__}",
            "judge_attempts": int(getattr(self.generator, "judge_attempts", 0)),
        }
        for name in ("base_guide", "reference_guide"):
            path = getattr(self.generator, f"{name}_path", None)
            if path:
                guide = load_markdown_guide(path)
                revision[f"{name}_sha256"] = hashlib.sha256(guide.encode("utf-8")).hexdigest()
        return revision

    def build_h3_prompt(self, *, recovery: Any = None, **kwargs: Any) -> dict[str, Any]:
        result = {}
        try:
            result = self._build_h3_prompt(recovery=recovery, **kwargs)
            self._validate_final_result(result, segment=kwargs["segment"], mode=kwargs.get("mode", "ref"))
        except PromptContractError as exc:
            if not self.allow_fallback:
                exc.candidate_result = result or getattr(exc, "candidate_result", {})
                raise
            result = result or getattr(exc, "candidate_result", {}) or {}
            reasons = [issue.code for issue in exc.issues]
            result.setdefault("prompt", "")
            result["prompt_contract"] = {"valid": False}
            result["validation_audio_delivery"] = (kwargs.get("global_context") or {}).get("h3_audio_delivery")
            result["dspy_error"] = "; ".join(reasons)
            result["readiness"] = recovery.finish("blocked", reasons, stage="validation") if recovery is not None else {
                "status": "blocked", "stage": "validation", "reason_codes": reasons,
                "policy_version": 1, "attempt_revision": 0, "attempts": []}
            return result
        result["prompt_contract"] = _valid_prompt_contract(result["prompt"])
        result["validation_audio_delivery"] = (kwargs.get("global_context") or {}).get("h3_audio_delivery")
        result["readiness"] = recovery.finish("ready", [], stage="validation") if recovery is not None else {
            "status": "ready", "stage": "validation", "reason_codes": [], "policy_version": 1,
            "attempt_revision": 0, "attempts": [],
        }
        return result

    @staticmethod
    def _reserve(recovery: Any, stage: str) -> None:
        if recovery is not None and not recovery.reserve(stage):
            raise PromptContractError([PromptContractIssue(
                "h3.recovery.exhausted", stage, "This input revision has already reserved this attempt")])

    @staticmethod
    def _validate_final_result(result: dict, *, segment: dict, mode: str) -> None:
        sections = result.get("sections") or {}
        prompt = str(result.get("prompt") or "")
        if sections.get("h3_sections") is not None:
            plan = H3PromptSections.model_validate(sections["h3_sections"]).to_plan()
            phases = _normalize_relay_segments(segment)
            references = result.get("references") or []
            bindings = _audio_subject_bindings(segment.get("references") or {}, available_stems={
                str(ref.get("name") or "") for ref in references if ref.get("kind") == "audio"})
            _stamp_relay_speaker_binding(phases, bindings)
            count, subject = _relay_vocal_binding(phases, bindings)
            issues = validate_h3_prompt_contract(prompt, mode=mode, plan=plan, reference_metadata=references,
                duration_seconds=segment.get("duration_seconds", segment.get("duration")),
                expected_vocal_events=count if phases else None, bound_vocal_subject=subject)
            issues.extend(validate_performance_phases(phases, prompt))
            if phases:
                def words(value):
                    return re.findall(r"\w+", value.casefold())
                from feverslop.prompting.deterministic_h3_compiler import _relay_vocal_content
                expected = [words(_relay_vocal_content(p)) for p in phases if _relay_vocal_content(p)]
                actual = [words(re.sub(r"^\[[^]]+\]\s*", "", text.strip()))
                          for text in re.findall(r"<d>(.*?)</d>", prompt, re.DOTALL)]
                if [word for event in actual for word in event] != [word for event in expected for word in event]:
                    issues.append(PromptContractIssue("h3.performance.lyrics_mismatch", "dialogue", "Final words differ from performance evidence"))
            for index, fact in enumerate(segment.get("locked_facts") or []):
                if str(fact.get("value") or "") not in prompt:
                    issues.append(PromptContractIssue("h3.fact.missing", f"locked_facts[{index}]", "Explicit locked fact is absent"))
        elif sections.get("shots") and sections.get("facts"):
            from feverslop.prompting.dspy_h3_models import CreativeShotPayload
            raw_facts = sections["facts"]
            issues = validate_prompt_contract(prompt, facts=raw_facts if isinstance(raw_facts, LockedSceneFacts) else LockedSceneFacts.from_dict(raw_facts),
                shots=[CreativeShotPayload.model_validate(s) for s in sections["shots"]],
                shot_windows=sections.get("shot_windows") or {}, references=sections.get("references") or {})
        else:
            issues = [PromptContractIssue("h3.plan.missing", "sections", "A structured creative plan is required for semantic validation")]
        if issues:
            raise PromptContractError(issues)

    def _build_h3_prompt(
        self,
        *,
        segment: dict[str, Any],
        concept: str,
        scene_details: dict[str, Any],
        global_context: dict[str, Any],
        mode: str = "ref",
        video_type: str = "music_video",
        audio_paths: dict[str, Path] | None = None,
        reference_root: Path | None = None,
        structured_sections: dict[str, Any] | None = None,
        recovery: Any = None,
    ) -> dict[str, Any]:
        if structured_sections is not None:
            return self._build_structured_prompt(
                mode=mode,
                segment=segment,
                concept=concept,
                global_context=global_context,
                sections=structured_sections,
            )
        audio_delivery = H3AudioDelivery.from_context(
            global_context.get("h3_audio_delivery"),
        )
        references, images = _scene_references(
            segment,
            audio_paths,
            reference_root or self.reference_root,
            mode=mode,
            audio_delivery=audio_delivery,
        )
        references = self._resolve_audio_delivery(references, audio_delivery, segment, reference_root)
        raw_bindings = _audio_subject_bindings(
            segment.get("references") or {},
            available_stems={str(reference["name"]) for reference in references if reference["kind"] == "audio"},
        )
        audio_subject_bindings = [
            AudioSubjectBinding(
                audio_label=next(reference["label"] for reference in references if reference["kind"] == "audio" and reference["name"] == stem),
                stem=stem,
                subject_label=binding["subject_label"],
                speaker_id=binding["speaker_id"] or None,
            )
            for stem, binding in raw_bindings.items()
        ]
        relay_segments = _normalize_relay_segments(segment)
        _stamp_relay_speaker_binding(relay_segments, raw_bindings)
        performance_issues = validate_performance_phases(relay_segments)
        if performance_issues:
            raise PromptContractError(performance_issues)
        directive_plan = subject_directives_from_scene(segment)
        generator_references = [dict(reference) for reference in references]
        directing_lines = [
            f"{key.replace('_', ' ').title()}: {str(scene_details[key]).strip()}"
            for key in ("camera_motion", "character_motion", "spatial_relations")
            if str(scene_details.get(key) or "").strip()
        ]
        user_prompt = str(concept or "").strip()
        creative_prompt = str(segment.get("h3_creative_prompt") or "").strip()
        if creative_prompt:
            user_prompt = (
                f"{user_prompt}\n\nExisting backend-neutral scene motion prompt:\n"
                f"{creative_prompt}"
            ).strip()
        if directive_plan is not None:
            user_prompt = f"{user_prompt}\n\n{project_directives_to_prompt(directive_plan)}".strip()
        if directing_lines:
            user_prompt = f"{user_prompt}\n\nScene-specific directing instructions:\n" + "\n".join(directing_lines)
        resolved_root = reference_root or self.reference_root
        if resolved_root is not None:
            for reference in generator_references:
                if reference["kind"] != "picture":
                    continue
                source = Path(reference["source"])
                image_path = source if source.is_absolute() else Path(resolved_root) / source
                if image_path.is_file():
                    reference["source"] = str(image_path)
        request = {
            "mode": mode,
            "video_type": video_type,
            "duration_seconds": segment.get("duration") or segment.get("duration_seconds"),
            "user_prompt": user_prompt,
            "source_language": str(global_context.get("language") or "").strip(),
            "notes": json.dumps(compact_planning_payload({
                "scene": segment,
                "scene_details": scene_details,
                "global_context": global_context,
                "source_language": str(global_context.get("language") or "").strip(),
                "language_policy": (
                    "Use the supplied source language for lyric/dialogue labels. "
                    "Preserve lyric text verbatim and do not infer language from proper names, "
                    "fantasy names, or isolated tokens."
                ),
            }), ensure_ascii=False),
            "references": generator_references,
            "images": images,
            "relay_segments": relay_segments,
            "strict_fidelity": True,
            "_section_only": True,
            "audio_subject_bindings": audio_subject_bindings,
        }
        has_reused_audio_reference = any(
            reference.get("kind") == "audio" and reference.get("role") == "audio_reuse"
            for reference in references
        )
        if (
            audio_delivery.conditions_generation
            and any(reference.get("kind") == "audio" for reference in references)
        ):
            # Reused scene/song audio is already supplied to H3 as <Audio N>.
            # The selected workflow supplies it as a masked audio latent and
            # carries it into the result, not as audience-only score to invent.
            request["music_intent"] = MusicIntent.NONE.value
        elif audio_paths or has_reused_audio_reference:
            # Legacy callers without a workflow delivery contract preserve the
            # earlier safe behavior. Production H3 runs always pass the contract.
            request["music_intent"] = MusicIntent.NONE.value
        # Resolve immutable inputs before the planner runs. Creative planning
        # may add actions, but it must not be the authority for scene facts.
        facts = locked_scene_facts_from_scene(segment)
        generated = None
        current_plan = None
        self._reserve(recovery, "generate")
        try:
            generated = self.generator(request)
            if hasattr(generated, "plan"):
                # Generation is intentionally single-pass. A scene must fall
                # back or retain an advisory BAD verdict rather than entering a
                # retry loop that can stall a long render batch.
                max_attempts = 2
                judge_attempts = []
                contract_repaired = False
                current_plan = generated.plan
                speaker_bindings = _speaker_bindings_for_compile(
                    segment=segment,
                    references=references,
                    stored_bindings=[
                        binding.model_dump()
                        for binding in request.get("audio_subject_bindings") or ()
                    ],
                )
                # The relay marks a window "singing" only when it carries lyrics,
                # and the compiler emits exactly one dialogue event per sung
                # window. Both counts must agree, and a bound vocal stem must be
                # anchored to its visible subject rather than an audible voice.
                relay_vocal_events, bound_vocal_subject = _relay_vocal_binding(relay_segments, raw_bindings)
                for attempt in range(max_attempts):
                    normalized_plan = _normalize_plan_audio_usage(
                        current_plan,
                        references,
                        audio_delivery=audio_delivery,
                    )
                    normalized_plan = plan_with_authoritative_relay(
                        normalized_plan,
                        request.get("relay_segments") or (),
                        language=str(request.get("source_language") or "English"),
                        speaker_bindings=speaker_bindings,
                    )
                    sections = H3PromptSections.from_plan(normalized_plan)
                    plan = sections.to_plan()
                    shots = creative_shots_from_plan(plan)
                    windows = {}
                    references_by_shot = {}
                    for shot, creative in zip(plan.shots, shots, strict=True):
                        start = float(shot.start_seconds or 0.0)
                        end = float(shot.end_seconds or segment.get("duration") or segment.get("duration_seconds") or (start + 1.0))
                        if end <= start:
                            end = start + 1.0
                        windows[creative.shot_id] = (start, end)
                        references_by_shot[creative.shot_id] = list(shot.reference_labels)
                    try:
                        prompt = DeterministicH3Compiler().compile(
                            mode=mode, plan=plan, facts=facts, shots=shots,
                            shot_windows=windows, references=references_by_shot,
                            prepared_reference_labels=[reference["label"] for reference in references],
                            reference_metadata=references,
                            duration_seconds=float(segment.get("duration") or segment.get("duration_seconds") or 0) or None,
                            dialogue_language=str(request.get("source_language") or "English"),
                            relay_segments=request.get("relay_segments") or (),
                            speaker_bindings=speaker_bindings,
                        )
                        self._validate_final_result({"prompt": prompt, "references": references,
                            "sections": {"h3_sections": sections.model_dump()}}, segment=segment, mode=mode)
                        shape_issues = []
                    except PromptContractError as exc:
                        shape_issues = list(exc.issues)
                    if shape_issues:
                        if attempt + 1 < max_attempts:
                            repair_request = dict(request)
                            repair_request["notes"] = (
                                f"{request['notes']}\n\n"
                                "Repair the typed H3 plan so the deterministic compiler satisfies "
                                "these contract requirements: "
                                + ", ".join(issue.code for issue in shape_issues)
                                + ". Preserve all locked facts, references, relay timing, and bindings."
                            )
                            self._reserve(recovery, "repair")
                            repaired = self.generator(repair_request)
                            repaired_plan = getattr(repaired, "plan", None)
                            if repaired_plan is not None:
                                current_plan = repaired_plan
                                contract_repaired = True
                                continue
                        raise PromptContractError(shape_issues)
                    result = {
                        "prompt": prompt,
                        "references": references,
                        "h3_audio_sources": [ref["audio_delivery"] for ref in references if "audio_delivery" in ref],
                        "prompt_contract": _valid_prompt_contract(prompt),
                        "sections": {
                            "h3_sections": sections.model_dump(),
                            "facts": facts.to_dict(),
                            "shots": [shot.model_dump() for shot in shots],
                            "shot_windows": {key: list(value) for key, value in windows.items()},
                            "references": references_by_shot,
                            "speaker_bindings": speaker_bindings,
                        },
                        "continuation_intents": [asdict(intent) for intent in sections.continuation_intents],
                        "prompt_provenance": {
                            "compiler": H3_COMPILER_NAME,
                            "compiler_version": H3_COMPILER_VERSION,
                            "source": (
                                "dspy_contract_repair"
                                if contract_repaired else "dspy_section_plan"
                            ),
                        },
                    }
                    judge_compiled = getattr(self.generator, "judge_compiled_prompt", None)
                    judged = judge_compiled(
                        request=request, plan=plan, references=references, final_prompt=prompt,
                    ) if callable(judge_compiled) else None
                    if judged is not None:
                        judge_attempts.append(judged.model_dump())
                        result["prompt_judge"] = judged.model_dump()
                        result["prompt_judge_attempts"] = list(judge_attempts)
                    # The judge is deliberately advisory.  Its full verdict and optional
                    # suggested prompt are persisted above, but never trigger another LLM
                    # call or replace the compiled prompt.
                    break
                return result
            prompt = getattr(generated, "rendered_prompt", None)
            if not prompt and isinstance(generated, dict):
                prompt = generated.get("rendered_prompt") or generated.get("prompt")
            if not prompt:
                raise ValueError("DSPy generator returned no rendered prompt")
        except Exception as exc:
            safe_error = _safe_error_message(exc)
            if not self.allow_fallback:
                raise RuntimeError(f"DSPy H3 generation failed: {safe_error}") from exc
            self._reserve(recovery, "fallback")
            try:
                result = self._deterministic_fallback(
                    mode=mode, segment=segment, concept=concept, facts=facts,
                    references=references, plan=current_plan, global_context=global_context,
                )
            except Exception as fallback_error:
                if current_plan is not None:
                    fallback_error.candidate_result = {"references": references, "sections": {
                        "h3_sections": H3PromptSections.from_plan(current_plan).model_dump(), "facts": facts.to_dict()}}
                raise
            result["dspy_error"] = safe_error
            return result
        # DSPy is solely responsible for the guide-conformant prompt. Do not
        # append or repair deterministic prose after generation.
        prompt_parts = [str(prompt).strip()]
        # Preserve raw legacy output for diagnostics; the shared final validator
        # blocks it when no structured plan supports semantic verification.
        result = {
            "prompt": "\n\n".join(part for part in prompt_parts if part),
            "references": references,
            "h3_audio_sources": [ref["audio_delivery"] for ref in references if "audio_delivery" in ref],
        }
        if directive_plan is not None:
            result["subject_directives"] = directive_plan.to_dict()
        if segment.get("performance_timing"):
            result["performance_timing"] = segment["performance_timing"]
        if isinstance(generated, dict) and generated.get("dspy_error"):
            fallback_shape_issues = validate_h3_prompt_shape(result["prompt"], mode=mode)
            if fallback_shape_issues:
                raise RuntimeError(
                    "deterministic H3 fallback violates the guide shape: "
                    + "; ".join(issue.code for issue in fallback_shape_issues),
                )
            result["prompt_contract"] = _valid_prompt_contract(result["prompt"])
            result["dspy_error"] = generated["dspy_error"]
            result["prompt_provenance"] = {
                "compiler": H3_COMPILER_NAME,
                "compiler_version": H3_COMPILER_VERSION,
                "source": "deterministic_fallback",
            }
        judge = getattr(generated, "judge", None)
        if judge is not None:
            result["prompt_judge"] = judge.model_dump()
        judge_attempts = getattr(generated, "judge_attempts", None)
        if judge_attempts:
            result["prompt_judge_attempts"] = [item.model_dump() for item in judge_attempts]
        return result

    def _deterministic_fallback(
        self, *, mode, segment, concept, facts, references, plan, global_context,
    ) -> dict[str, Any]:
        """Repair only deterministic fields of the last structured creative plan."""
        if plan is None:
            raise PromptContractError([PromptContractIssue(
                "h3.fallback.plan_missing", "fallback", "No structured creative plan survived generation")])
        if not str(plan.style_opening or "").strip():
            plan = plan.model_copy(update={
                "style_opening": "Live-action cinematic imagery preserves the planned composition and scene facts.",
            })
        result = self._build_structured_prompt(mode=mode, segment=segment, concept=concept,
            global_context=global_context, judge=False, sections={
                "facts": facts.to_dict(), "h3_sections": H3PromptSections.from_plan(plan).model_dump(),
                "resolved_references": references,
            })
        result["prompt_provenance"]["source"] = "deterministic_fallback"
        return result

    def _report_warning(self, message: str, *, title: str) -> None:
        warning = getattr(self.generator, "_warning", None)
        if callable(warning):
            warning(message, title=title)

    def _build_structured_prompt(
        self,
        *,
        mode: str,
        segment: dict[str, Any],
        concept: str,
        global_context: dict[str, Any],
        sections: dict[str, Any],
        judge: bool = True,
    ) -> dict[str, Any]:
        """Compile planner-owned sections without invoking the legacy prose generator.

        The planner may return JSON-compatible dictionaries; validation is kept at
        this boundary so backend labels, timing, and locked facts cannot bypass the
        deterministic compiler contract.
        """
        raw_facts = sections.get("facts")
        facts = (
            raw_facts
            if isinstance(raw_facts, LockedSceneFacts)
            else LockedSceneFacts.from_dict(raw_facts)
        )
        raw_h3_sections = sections.get("h3_sections")
        if raw_h3_sections is not None:
            h3_sections = H3PromptSections.model_validate(raw_h3_sections)
            plan = h3_sections.to_plan()
            audio_delivery = H3AudioDelivery.from_context(
                global_context.get("h3_audio_delivery"),
            )
            resolved_references = _normalize_resolved_scene_references(
                list(sections.get("resolved_references") or []),
                audio_delivery=audio_delivery,
            )
            resolved_references = self._resolve_audio_delivery(resolved_references, audio_delivery, segment)
            plan = _normalize_plan_audio_usage(
                plan,
                resolved_references,
                audio_delivery=audio_delivery,
            )
            speaker_bindings = _speaker_bindings_for_compile(
                segment=segment,
                references=resolved_references,
                stored_bindings=list(sections.get("speaker_bindings") or ()),
            )
            raw_bindings = _audio_subject_bindings(
                segment.get("references") or {},
                available_stems={
                    str(reference.get("name") or "").strip()
                    for reference in resolved_references
                    if str(reference.get("kind") or "").casefold() == "audio"
                    and str(reference.get("name") or "").strip()
                },
            )
            relay_segments = _normalize_relay_segments(segment)
            _stamp_relay_speaker_binding(relay_segments, raw_bindings)
            relay_vocal_events, bound_vocal_subject = _relay_vocal_binding(relay_segments, raw_bindings)
            plan = plan_with_authoritative_relay(
                plan,
                relay_segments,
                language=str(global_context.get("language") or "English"),
                speaker_bindings=speaker_bindings,
            )
            shots = creative_shots_from_plan(plan)
            stored_windows = sections.get("shot_windows") or {}
            windows: dict[str, tuple[float, float]] = {}
            references_by_shot: dict[str, list[str]] = {}
            stored_references = sections.get("references") or {}
            duration = float(
                segment.get("duration") or segment.get("duration_seconds") or 0
            ) or None
            for planned, shot in zip(plan.shots, shots, strict=True):
                raw_window = stored_windows.get(shot.shot_id)
                start = float(
                    raw_window[0]
                    if isinstance(raw_window, (list, tuple)) and len(raw_window) == 2
                    else planned.start_seconds or 0.0
                )
                end = float(
                    raw_window[1]
                    if isinstance(raw_window, (list, tuple)) and len(raw_window) == 2
                    else planned.end_seconds or duration or (start + 1.0)
                )
                if end <= start:
                    end = start + 1.0
                windows[shot.shot_id] = (start, end)
                references_by_shot[shot.shot_id] = list(
                    stored_references.get(shot.shot_id) or planned.reference_labels
                )
            prompt = DeterministicH3Compiler().compile(
                mode=mode,
                plan=plan,
                facts=facts,
                shots=shots,
                shot_windows=windows,
                references=references_by_shot,
                prepared_reference_labels=[
                    str(reference["label"])
                    for reference in resolved_references
                    if isinstance(reference, dict) and reference.get("label")
                ],
                reference_metadata=resolved_references,
                duration_seconds=duration,
                dialogue_language=str(global_context.get("language") or "English"),
                relay_segments=relay_segments,
                speaker_bindings=speaker_bindings,
            )
            shape_issues = validate_h3_prompt_contract(
                prompt,
                mode=mode,
                plan=plan,
                reference_metadata=resolved_references,
                duration_seconds=duration,
                expected_vocal_events=relay_vocal_events,
                bound_vocal_subject=bound_vocal_subject,
            )
            if shape_issues:
                raise PromptContractError(shape_issues)
            result = {
                "prompt": prompt,
                "references": resolved_references,
                "h3_audio_sources": [ref["audio_delivery"] for ref in resolved_references if "audio_delivery" in ref],
                "prompt_contract": _valid_prompt_contract(prompt),
                "segment_id": segment.get("segment_id"),
                "sections": sections,
                "continuation_intents": [
                    asdict(intent) for intent in h3_sections.continuation_intents
                ],
                "prompt_provenance": {
                    "compiler": H3_COMPILER_NAME,
                    "compiler_version": H3_COMPILER_VERSION,
                    "source": "resumed_dspy_section_plan",
                },
            }
            judge_compiled = getattr(self.generator, "judge_compiled_prompt", None)
            if judge and callable(judge_compiled):
                judged = judge_compiled(
                    request={
                        "mode": mode,
                        "user_prompt": str(concept),
                        "duration_seconds": duration,
                        "strict_fidelity": True,
                    },
                    plan=plan,
                    references=resolved_references,
                    final_prompt=prompt,
                )
                if judged is not None:
                    result["prompt_judge"] = judged.model_dump()
                    result["prompt_judge_attempts"] = [judged.model_dump()]
            return result

        from feverslop.prompting.dspy_h3_models import CreativeShotPayload

        shots = [
            shot if isinstance(shot, CreativeShotPayload) else CreativeShotPayload.model_validate(shot)
            for shot in sections.get("shots") or []
        ]
        windows = sections.get("shot_windows") or {}
        references = sections.get("references") or {}
        prompt = DeterministicH3Compiler().compile(
            mode="base" if str(mode).lower() == "base" else "reference",
            facts=facts,
            shots=shots,
            shot_windows=windows,
            references=references,
        )
        return {
            "prompt": prompt,
            "references": sections.get("resolved_references") or [],
            "segment_id": segment.get("segment_id"),
            "sections": sections,
            "continuation_intents": list(sections.get("continuation_intents") or []),
            "prompt_provenance": {
                "compiler": H3_COMPILER_NAME,
                "compiler_version": H3_COMPILER_VERSION,
                "source": "structured_sections",
            },
        }

    def _prepare_recoverable_scene(
        self, *, segment, concept, details, global_context, mode, video_type, audio_paths,
        reference_root, structured_sections, checkpoint_store, checkpoint_input,
        recovery, cached, reuse_checkpoints,
    ):
        override = _h3_prompt_override(segment)
        stale_loader = getattr(checkpoint_store, "load_for_resume", None)
        stale = stale_loader(checkpoint_input) if callable(stale_loader) else None
        if cached is None and recovery is None and reuse_checkpoints and checkpoint_store is not None:
            loader = getattr(checkpoint_store, "load_advisory", checkpoint_store.load)
            checkpoint = loader(checkpoint_input)
            cached = checkpoint.generated if checkpoint is not None else None
        classifier = getattr(checkpoint_store, "invalidated_stages", None)
        if (cached is None and reuse_checkpoints and stale is not None
                and not stale.generated.get("readiness") and callable(classifier)
                and not classifier(checkpoint_input, stale)):
            cached = stale.generated
        if cached is not None and not override:
            try:
                self._validate_final_result(cached, segment=segment, mode=mode)
            except PromptContractError:
                pass
            else:
                result = dict(cached)
                if recovery is not None:
                    result["readiness"] = recovery.finish("ready", [], stage="validation")
                return result, "reused"
        status = "completed"
        stale_facts = (stale.generated.get("sections") or {}).get("facts") if stale is not None else None
        can_reuse_override_plan = (
            stale is not None and bool(stale.generated.get("sections"))
            and stale_facts == locked_scene_facts_from_scene(segment).to_dict()
            and callable(classifier)
            and not (classifier(checkpoint_input, stale) & {"locked_facts", "workflow", "all"})
            and stale.generated.get("validation_audio_delivery") == global_context.get("h3_audio_delivery")
        )
        if override and can_reuse_override_plan:
            result = dict(stale.generated)
        else:
            classifier = getattr(checkpoint_store, "invalidated_stages", None)
            if (structured_sections is None and stale is not None and reuse_checkpoints and callable(classifier)
                    and classifier(checkpoint_input, stale) == frozenset({"compiler"})
                    and stale.generated.get("sections")):
                structured_sections = {**stale.generated["sections"], "resolved_references": stale.generated.get("references") or []}
                status = "recompiled"
            build_kwargs = dict(segment=segment, concept=concept, scene_details=details,
                global_context=global_context, mode=mode, video_type=video_type,
                audio_paths=audio_paths, reference_root=reference_root, recovery=recovery)
            try:
                result = self.build_h3_prompt(**build_kwargs, structured_sections=structured_sections)
            except PromptContractError:
                if status != "recompiled":
                    raise
                result = self.build_h3_prompt(**build_kwargs)
                status = "regenerating"
            else:
                if status == "recompiled" and (result.get("readiness") or {}).get("status") == "blocked":
                    result = self.build_h3_prompt(**build_kwargs)
                    status = "regenerating"
        if override:
            result = {**result, "prompt": override, "prompt_provenance": {
                "source": "user_override", "compiler": H3_COMPILER_NAME, "compiler_version": H3_COMPILER_VERSION}}
            try:
                self._validate_final_result(result, segment=segment, mode=mode)
            except PromptContractError as exc:
                exc.candidate_result = result
                raise
            result["prompt_contract"] = _valid_prompt_contract(override)
            result["readiness"] = recovery.finish("ready", [], stage="validation") if recovery is not None else {"status": "ready", "reason_codes": []}
            status = "override"
        if (result.get("readiness") or {}).get("status") == "blocked":
            status = "blocked"
        return result, status

    def build_all_h3_prompts(
        self,
        *,
        stage1_segments: list[dict],
        concept_prompts: dict,
        scene_details: dict,
        global_context: dict,
        mode: str = "ref",
        video_type: str = "music_video",
        output_json_path: str | Path,
        artifact_store,
        audio_paths: dict[str, Path] | None = None,
        reference_root: Path | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        status_callback: Callable[[int, int, str], None] | None = None,
        warning_callback: Callable[..., None] | None = None,
        checkpoint_store: H3PromptCheckpointPort | None = None,
        generator_revision: dict[str, Any] | None = None,
        preserve_existing_aggregate: bool = False,
        reuse_checkpoints: bool = True,
        structured_sections_by_segment: dict[str, dict[str, Any]] | None = None,
    ) -> Path:
        set_warning_callback = getattr(self.generator, "set_warning_callback", None)
        if callable(set_warning_callback):
            set_warning_callback(warning_callback)
        results = []
        total = len(stage1_segments)
        from contextlib import nullcontext
        for current, segment in enumerate(stage1_segments, start=1):
            segment_id = segment["segment_id"]
            concept = concept_prompts.get(segment_id, "")
            if isinstance(concept, dict):
                concept = concept.get("concept", "")
            details = scene_details.get(segment_id, {})
            checkpoint_input = H3PromptCheckpointInput(
                scene_number=int(segment.get("scene") or segment.get("scene_number") or current),
                segment_id=str(segment_id), segment=segment, concept=str(concept), scene_details=details,
                global_context=global_context, mode=mode, video_type=video_type,
                audio_paths=audio_paths or {}, generator_revision=generator_revision or self.checkpoint_revision())
            session_factory = getattr(checkpoint_store, "recovery_session", None)
            if status_callback is not None:
                status_callback(current, total, "started")
            session_context = session_factory(checkpoint_input, replan=not reuse_checkpoints) if callable(session_factory) else nullcontext(None)
            with session_context as recovery:
                cached = recovery.saved_result if recovery is not None else None
                if cached is not None and (cached.get("readiness") or {}).get("status") == "blocked":
                    result = cached
                    status = "blocked"
                else:
                    try:
                        result, status = self._prepare_recoverable_scene(
                            segment=segment, concept=str(concept), details=details, global_context=global_context,
                            mode=mode, video_type=video_type, audio_paths=audio_paths, reference_root=reference_root,
                            structured_sections=(structured_sections_by_segment or {}).get(segment_id),
                            checkpoint_store=checkpoint_store, checkpoint_input=checkpoint_input,
                            recovery=recovery, cached=cached, reuse_checkpoints=reuse_checkpoints)
                    except Exception as exc:
                        reasons = ([issue.code for issue in exc.issues] if isinstance(exc, PromptContractError)
                                   else [str(getattr(exc, "code", "h3.preparation.failed"))])
                        result = dict(getattr(exc, "candidate_result", {}) or {})
                        result.update(prompt="", prompt_contract={"valid": False},
                            prompt_provenance={"compiler": H3_COMPILER_NAME, "compiler_version": H3_COMPILER_VERSION,
                                               "source": "blocked"})
                        result["readiness"] = recovery.finish("blocked", reasons, stage="preparation") if recovery is not None else {
                            "status": "blocked", "stage": "preparation", "reason_codes": reasons,
                            "policy_version": 1, "attempt_revision": 0, "attempts": []}
                        status = "blocked"
                    if checkpoint_store is not None:
                        checkpoint_store.save(checkpoint_input, result)
                if status == "blocked" and warning_callback is not None:
                    readiness = result.get("readiness") or {}
                    reasons = ", ".join(readiness.get("reason_codes") or ["h3.preparation.failed"])
                    attempts = ", ".join(a["stage"] for a in readiness.get("attempts") or []) or "none"
                    warning_callback(f"Scene {checkpoint_input.scene_number} blocked: {reasons}; reserved attempts: {attempts}. "
                        "Correct its inputs or explicitly replan this scene; unchanged resume spends no new attempts.",
                        title="H3 scene readiness")
                results.append({"segment_id": segment_id, **result})
            if progress_callback is not None:
                progress_callback(current, total)
            if status_callback is not None:
                status_callback(current, total, status)
        if preserve_existing_aggregate and Path(output_json_path).is_file():
            existing = artifact_store.read_json(output_json_path)
            if not isinstance(existing, list) or any(not isinstance(item, dict) for item in existing):
                raise ValueError(f"H3 prompt aggregate must be a list of objects: {output_json_path}")
            updates = {str(item.get("segment_id")): item for item in results}
            merged = []
            handled: set[str] = set()
            for item in existing:
                segment_id = str(item.get("segment_id"))
                merged.append(updates.get(segment_id, item))
                handled.add(segment_id)
            merged.extend(
                item for item in results
                if str(item.get("segment_id")) not in handled
            )
            results = merged
        return artifact_store.write_json(output_json_path, results)


def build_dspy_generator(llm: Any) -> Callable[[dict[str, Any]], Any]:
    """Create the complete planner/analyzer/renderer generator from dspy_prompt_test."""
    from feverslop.prompting.dspy_h3_generator import VideoPromptGenerator

    return VideoPromptGenerator(
        base_guide_path="minimax-h3-base.md",
        reference_guide_path="minimax-h3-references.md",
        llm=llm,
    )


def _valid_prompt_contract(prompt: str) -> dict[str, Any]:
    return {
        "valid": True,
        "validator": "minimax_h3_guide_contract",
        "compiler_version": H3_COMPILER_VERSION,
        "prompt_sha256": "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "issues": [],
    }


def _judge_feedback(judged: Any) -> str:
    parts = [str(issue).strip() for issue in judged.issues if str(issue).strip()]
    repair_instruction = str(judged.repair_instruction or "").strip()
    if repair_instruction:
        parts.append(repair_instruction)
    parts.extend(
        f"{issue.shot_id}.{issue.field}: {issue.repair_instruction}"
        for issue in judged.field_issues
    )
    return "; ".join(dict.fromkeys(parts)) or "the prompt did not satisfy the guide"


def _h3_prompt_override(segment: Mapping[str, Any]) -> str:
    """Return an opaque user prompt override without interpreting its format."""
    direct = segment.get("h3_prompt_override")
    if isinstance(direct, str) and direct.strip():
        return direct
    canonical = segment.get("canonical")
    roles = canonical.get("roles") if isinstance(canonical, Mapping) else None
    role = roles.get("h3.video") if isinstance(roles, Mapping) else None
    override = role.get("override") if isinstance(role, Mapping) else None
    value = override.get("value") if isinstance(override, Mapping) else None
    return value if isinstance(value, str) and value.strip() else ""


def _sanitize_judge_feedback(value: str) -> str:
    """Keep compiler-owned labels out of planner repair instructions."""
    return re.sub(
        r"<(Subject|Picture|Video|Audio)\s+\d+>",
        lambda match: {
            "subject": "the referenced subject",
            "picture": "the reference image",
            "video": "the reference video",
            "audio": "the reference audio",
        }[match.group(1).casefold()],
        str(value),
        flags=re.IGNORECASE,
    )
