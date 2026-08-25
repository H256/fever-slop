from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from feverslop.domain.performance_sync import select_performance_audio_paths
from feverslop.domain.h3_prompt_checkpoint import H3PromptCheckpointInput
from feverslop.prompting.dspy_h3_models import MusicIntent
from feverslop.prompting.guide_loader import load_markdown_guide
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


def _audio_subject_bindings(references: dict[str, Any]) -> dict[str, dict[str, str]]:
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
    for stem, value in raw.items():
        stem_name = str(stem).strip()
        if not stem_name or not isinstance(value, dict):
            raise ValueError("each audio subject binding requires a stem and object value")
        if stem_name == "full_mix":
            raise ValueError("full_mix is global audio and cannot bind to a subject")
        subject_id = str(value.get("subject_id") or value.get("subject") or "").strip()
        subject_label = str(value.get("subject_label") or actor_labels.get(subject_id) or "").strip()
        if not subject_label:
            raise ValueError(f"audio binding for {stem_name!r} has no known subject")
        if subject_label in seen_subjects:
            raise ValueError(f"multiple audio stems bind to subject {subject_label}")
        seen_subjects.add(subject_label)
        speaker_id = str(value.get("speaker_id") or "").strip()
        if stem_name == "vocals" and not speaker_id:
            raise ValueError("vocal audio binding requires speaker_id")
        result[stem_name] = {
            "subject_label": subject_label,
            "speaker_id": speaker_id,
            "subject_id": subject_id,
        }
    return result


def _audio_description(name: str, binding: dict[str, str] | None) -> str:
    if name == "full_mix":
        return "full_mix - original song for beat and rhythm continuity"
    if binding is None:
        return f"{name} stem; no subject binding was supplied"
    speaker = f" ({binding['speaker_id']})" if binding.get("speaker_id") else ""
    return f"{name} stem bound to {binding['subject_label']}{speaker}"


def _scene_references(
    segment: dict[str, Any],
    audio_paths: dict[str, Path] | None,
    reference_root: Path | None,
    mode: str = "r2v",
) -> tuple[list[dict[str, str]], list[Path]]:
    references = segment.get("references") or {}
    selected_audio_paths = (
        select_performance_audio_paths(segment, audio_paths, max_stems=2)
        if audio_paths
        else {}
    )
    relay = (segment.get("ltx") or {}).get("prompt_relay") or segment.get("prompt_relay") or []
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
        copy_mode = "fully_copy" if "full_mix" in tag.casefold() else "partially_copy"
        description = tag or _audio_description(path.stem, audio_bindings.get(path.stem))
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
        ) | {"copy_mode": "fully_copy" if name == "full_mix" else "partially_copy"})

    for reference in pending_audio_references:
        add_reference(reference)

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
    relay = (segment.get("ltx") or {}).get("prompt_relay") or segment.get("prompt_relay") or []
    if not relay:
        return []
    fps = float(segment.get("fps") or 24)
    duration_value = segment.get("duration_seconds") or segment.get("duration")
    duration = float(duration_value) if duration_value is not None else None
    shots = []
    for index, item in enumerate(relay, start=1):
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
        source_prompt = str(item.get("source_prompt") or "").strip()
        if source_prompt:
            shot["source_prompt"] = source_prompt
        shots.append(shot)
    return shots


def _format_relay_shots(shots: list[dict[str, Any]]) -> str:
    if not shots:
        return ""
    lines = ["Temporal shot directions:"]
    for shot in shots:
        state = f" ({shot['state']})" if shot.get("state") else ""
        lines.append(
            f"[Shot {shot['shot']}, {shot['start_seconds']:.2f}-{shot['end_seconds']:.2f}sec]"
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
    ):
        self.generator = generator
        self.reference_root = reference_root
        self.allow_fallback = allow_fallback

    def checkpoint_revision(self) -> dict[str, Any]:
        revision: dict[str, Any] = {
            "contract": 1,
            "generator": f"{type(self.generator).__module__}.{type(self.generator).__qualname__}",
            "judge_attempts": int(getattr(self.generator, "judge_attempts", 0)),
        }
        for name in ("base_guide", "reference_guide"):
            path = getattr(self.generator, f"{name}_path", None)
            if path:
                guide = load_markdown_guide(path)
                revision[f"{name}_sha256"] = hashlib.sha256(guide.encode("utf-8")).hexdigest()
        return revision

    def build_h3_prompt(
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
    ) -> dict[str, Any]:
        references, images = _scene_references(
            segment,
            audio_paths,
            reference_root or self.reference_root,
            mode=mode,
        )
        relay_segments = _normalize_relay_segments(segment)
        directive_plan = subject_directives_from_scene(segment)
        generator_references = [dict(reference) for reference in references]
        directing_lines = [
            f"{key.replace('_', ' ').title()}: {str(scene_details[key]).strip()}"
            for key in ("camera_motion", "character_motion", "spatial_relations")
            if str(scene_details.get(key) or "").strip()
        ]
        user_prompt = str(concept or "").strip()
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
            "notes": json.dumps({
                "scene": segment,
                "scene_details": scene_details,
                "global_context": global_context,
                "source_language": str(global_context.get("language") or "").strip(),
                "language_policy": (
                    "Use the supplied source language for lyric/dialogue labels. "
                    "Preserve lyric text verbatim and do not infer language from proper names, "
                    "fantasy names, or isolated tokens."
                ),
            }, ensure_ascii=False),
            "references": generator_references,
            "images": images,
            "relay_segments": relay_segments,
            "strict_fidelity": True,
        }
        has_reused_audio_reference = any(
            reference.get("kind") == "audio" and reference.get("role") == "audio_reuse"
            for reference in references
        )
        if audio_paths or has_reused_audio_reference:
            # Reused scene/song audio is already supplied to H3 as <Audio N>.
            # It is not an additional audience-only score to invent. This must
            # be explicit even when the audio reference came from segment
            # metadata rather than the managed audio_paths argument.
            request["music_intent"] = MusicIntent.NONE.value
        generated = None
        try:
            generated = self.generator(request)
            prompt = getattr(generated, "rendered_prompt", None)
            if not prompt and isinstance(generated, dict):
                prompt = generated.get("rendered_prompt") or generated.get("prompt")
            if not prompt:
                raise ValueError("DSPy generator returned no rendered prompt")
        except Exception as exc:
            safe_error = _safe_error_message(exc)
            if not self.allow_fallback:
                raise RuntimeError(f"DSPy H3 generation failed: {safe_error}") from exc
            # Keep the legacy fallback for projects that cannot use DSPy, but
            # expose the reason to callers instead of silently claiming that a
            # structured DSPy prompt was generated.
            prompt = concept
            if isinstance(generated, dict):
                generated.setdefault("dspy_error", safe_error)
            else:
                generated = {"dspy_error": safe_error}
        # DSPy is solely responsible for the guide-conformant prompt. Do not
        # append or repair deterministic prose after generation.
        prompt_parts = [str(prompt).strip()]
        # The final prompt is judged by the DSPy prompt judge. Do not apply a
        # second deterministic semantic gate here: a rejected prompt must be
        # persisted with the judge result so a long batch can continue.
        result = {
            "prompt": "\n\n".join(part for part in prompt_parts if part),
            "references": references,
        }
        if directive_plan is not None:
            result["subject_directives"] = directive_plan.to_dict()
        if segment.get("performance_timing"):
            result["performance_timing"] = segment["performance_timing"]
        if isinstance(generated, dict) and generated.get("dspy_error"):
            result["dspy_error"] = generated["dspy_error"]
        judge = getattr(generated, "judge", None)
        if judge is not None:
            result["prompt_judge"] = judge.model_dump()
        judge_attempts = getattr(generated, "judge_attempts", None)
        if judge_attempts:
            result["prompt_judge_attempts"] = [item.model_dump() for item in judge_attempts]
        return result

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
    ) -> Path:
        set_warning_callback = getattr(self.generator, "set_warning_callback", None)
        if callable(set_warning_callback):
            set_warning_callback(warning_callback)
        results = []
        total = len(stage1_segments)
        for current, segment in enumerate(stage1_segments, start=1):
            segment_id = segment["segment_id"]
            if status_callback is not None:
                status_callback(current, total, "started")
            concept = concept_prompts.get(segment_id, "")
            if isinstance(concept, dict):
                concept = concept.get("concept", "")
            details = scene_details.get(segment_id, {})
            checkpoint_input = None
            checkpoint = None
            if checkpoint_store is not None:
                checkpoint_input = H3PromptCheckpointInput(
                    scene_number=int(segment.get("scene") or segment.get("scene_number") or current),
                    segment_id=str(segment_id),
                    segment=segment,
                    concept=str(concept),
                    scene_details=details,
                    global_context=global_context,
                    mode=mode,
                    video_type=video_type,
                    audio_paths=audio_paths or {},
                    generator_revision=generator_revision or {},
                )
                if reuse_checkpoints:
                    checkpoint = checkpoint_store.load(checkpoint_input)
            if checkpoint is not None:
                result = checkpoint.generated
            else:
                result = self.build_h3_prompt(
                    segment=segment,
                    concept=str(concept),
                    scene_details=details,
                    global_context=global_context,
                    mode=mode,
                    video_type=video_type,
                    audio_paths=audio_paths,
                    reference_root=reference_root,
                )
                if checkpoint_store is not None and checkpoint_input is not None:
                    checkpoint_store.save(checkpoint_input, result)
            results.append({"segment_id": segment_id, **result})
            if progress_callback is not None:
                progress_callback(current, total)
            if status_callback is not None:
                status_callback(current, total, "completed")
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
