from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from feverslop.prompting.dspy_h3_models import MusicIntent
from feverslop.domain.performance_sync import (
    select_performance_audio_paths,
    visible_performance_roles,
)


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
        result.append(reference)
        if image_path is not None and image_path.is_file():
            images.append(image_path)

    actor_paths = references.get("actor_sheet_paths") or references.get("actor_msr_paths") or []
    actor_ids = references.get("actor_ids") or []
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
        add_reference(_reference(
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
        ), image_path)

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
                or "Location"
            ),
            description=(
                str(
                    location_description.get("visual_description")
                    or location_description.get("image_prompt")
                    or ""
                ).strip()
                or ("" if image_path.is_file() else None)
            ),
            role="environment",
        ), image_path)

    for index, path_value in enumerate(references.get("reference_image_paths") or []):
        path = Path(path_value)
        image_path = path if path.is_absolute() or reference_root is None else reference_root / path
        role = "subject"
        if mode == "i2v" and index == 0:
            role = "first_frame"
        elif mode == "fl2v" and index == 0:
            role = "first_frame"
        elif mode == "fl2v" and index == 1:
            role = "last_frame"
        elif mode == "l2v" and index == 0:
            role = "last_frame"
        add_reference(_reference(
            label=f"<Picture {sum(kind == 'picture' for kind in seen['picture']) + 1}>",
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
        pending_audio_references.append(_reference(
            label=f"<Audio {len([ref for ref in result if ref['kind'] == 'audio']) + 1}>",
            source=path,
            kind="audio",
            name=path.stem,
            description=tag or "Use this synchronized reference for the scene's audio behavior.",
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
            description="Use this synchronized stem for the scene's audio behavior.",
            role="audio_reuse",
        ) | {"copy_mode": "fully_copy" if name == "full_mix" else "partially_copy"})

    for reference in pending_audio_references:
        reference["label"] = f"<Audio {len([ref for ref in result if ref['kind'] == 'audio']) + 1}>"
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


_H3_PROMPT_SECTIONS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)
_H3_SECTION_PATTERN = re.compile(
    r"^(subject_definitions|summary|retention_analysis|detailed_description|"
    r"overall_soundscape|non_diegetic_music):",
    re.MULTILINE,
)


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
            f"{state} {shot['prompt']}"
        )
        source_prompt = shot.get("source_prompt")
        if source_prompt and source_prompt != shot.get("prompt"):
            lines.append(f"Required action and props to preserve: {source_prompt}")
    return "\n".join(lines)


def _format_performance_timing(segment: dict[str, Any]) -> str:
    timing = segment.get("performance_timing") or {}
    beats = timing.get("beats") or []
    if not beats:
        return ""
    beat_times = ", ".join(f"{float(beat['time_seconds']):.2f}s" for beat in beats)
    downbeats = ", ".join(
        f"{float(beat['time_seconds']):.2f}s" for beat in beats if beat.get("downbeat")
    ) or "none in this shot"
    lines = [
        f"Performance timing (scene-local): BPM {float(timing.get('bpm') or 0):g}; "
        f"beats at {beat_times}; downbeats at {downbeats}."
    ]
    roles = visible_performance_roles(segment)
    if "drums" in roles:
        lines.append(
            "Drummer motion: prepare before each event, make stick contact exactly on each listed beat, "
            "emphasize downbeats, then rebound immediately; avoid random arm flailing."
        )
    if "bass" in roles:
        lines.append("Bassist motion: finger or pick contact lands on listed beats, with restrained release between notes.")
    if "other" in roles:
        lines.append("Guitarist motion: pick contact and chord changes land on listed beats, emphasizing downbeats.")
    return "\n".join(lines)


def _repair_audio_references(prompt: str, references: list[dict[str, str]]) -> str:
    """Restore audio_reuse semantics omitted by a sectioned DSPy H3 prompt."""
    audio_references = [
        reference
        for reference in references
        if reference["kind"] == "audio" and reference["role"] == "audio_reuse"
    ]
    if not audio_references:
        return prompt

    matches = list(_H3_SECTION_PATTERN.finditer(prompt))
    if [match.group(1) for match in matches] != list(_H3_PROMPT_SECTIONS):
        return prompt

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(prompt)
        sections[match.group(1)] = prompt[match.end():next_start]
    labels = [reference["label"] for reference in audio_references]
    label_text = " and ".join(labels)

    def append(section: str, text: str, *, inline: bool = False) -> None:
        if text not in sections[section]:
            separator = " " if inline else "\n"
            sections[section] = f"{sections[section].rstrip()}{separator}{text}\n\n"

    for reference in audio_references:
        label = reference["label"]
        definition = f"{label} is the synchronized {reference['name']} audio reference and is reused for the scene."
        definition_pattern = re.compile(rf"(?m)^{re.escape(label)}[^\n]*(?:\n|$)")
        sections["subject_definitions"] = definition_pattern.sub("", sections["subject_definitions"])
        append("subject_definitions", definition)
        retention_pattern = re.compile(
            rf"(?m)^{re.escape(label)}(?:\s+\([^\n]*?\))?:[^\n]*(?:\n|$)"
        )
        copy_mode = reference.get("copy_mode", "partially_copy")
        retention_lines = retention_pattern.findall(sections["retention_analysis"])
        if len(retention_lines) != 1 or copy_mode not in retention_lines[0]:
            sections["retention_analysis"] = retention_pattern.sub("", sections["retention_analysis"])
            append(
                "retention_analysis",
                f"{label}: {copy_mode} - the synchronized {reference['name']} audio is reused for this scene.",
            )

    if "audio reuse" not in sections["summary"].lower():
        summary = sections["summary"]
        task_type = re.search(r"\[([^\]]+)\]", summary)
        if task_type:
            sections["summary"] = (
                f"{summary[:task_type.start()]}[{task_type.group(1)} + audio reuse]"
                f"{summary[task_type.end():]}"
            )
        else:
            sections["summary"] = f"\n[audio reuse]{summary}"
    if not all(label in sections["summary"] for label in labels):
        append("summary", f"Audio reuse follows {label_text}.", inline=True)

    behavior = f"The synchronized audio behavior follows {label_text}."
    for section in ("detailed_description", "overall_soundscape"):
        if not all(label in sections[section] for label in labels):
            append(section, behavior, inline=True)

    non_diegetic = f"The synchronized audio references are scene inputs, not non-diegetic music: {label_text}."
    if non_diegetic not in sections["non_diegetic_music"]:
        if sections["non_diegetic_music"].strip() == "N/A":
            sections["non_diegetic_music"] = "\n"
        append("non_diegetic_music", non_diegetic, inline=True)

    return "".join(f"{section}:{sections[section]}" for section in _H3_PROMPT_SECTIONS).strip()


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
        append_relay_prompt: bool = True,
    ) -> dict[str, Any]:
        references, images = _scene_references(
            segment,
            audio_paths,
            reference_root or self.reference_root,
            mode=mode,
        )
        relay_segments = _normalize_relay_segments(segment)
        generator_references = [dict(reference) for reference in references]
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
            "user_prompt": concept,
            "notes": json.dumps({
                "scene": segment,
                "scene_details": scene_details,
                "global_context": global_context,
            }, ensure_ascii=False),
            "references": generator_references,
            "images": images,
            "relay_segments": relay_segments,
            "strict_fidelity": True,
        }
        if audio_paths:
            # The song/stems are supplied as referenced scene audio. They are
            # not an audience-only score generated by the H3 prompt writer.
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
        rendered_prompt = _repair_audio_references(str(prompt).strip(), references)
        relay_prompt = _format_relay_shots(relay_segments) if append_relay_prompt else ""
        performance_prompt = _format_performance_timing(segment)
        prompt_parts = [rendered_prompt, relay_prompt, performance_prompt]
        result = {
            "prompt": "\n\n".join(part for part in prompt_parts if part),
            "references": references,
        }
        if segment.get("performance_timing"):
            result["performance_timing"] = segment["performance_timing"]
        if isinstance(generated, dict) and generated.get("dspy_error"):
            result["dspy_error"] = generated["dspy_error"]
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
    ) -> Path:
        results = []
        total = len(stage1_segments)
        for current, segment in enumerate(stage1_segments, start=1):
            segment_id = segment["segment_id"]
            if status_callback is not None:
                status_callback(current, total, "started")
            concept = concept_prompts.get(segment_id, "")
            if isinstance(concept, dict):
                concept = concept.get("concept", "")
            result = self.build_h3_prompt(
                segment=segment,
                concept=str(concept),
                scene_details=scene_details.get(segment_id, {}),
                global_context=global_context,
                mode=mode,
                video_type=video_type,
                audio_paths=audio_paths,
                reference_root=reference_root,
            )
            results.append({"segment_id": segment_id, **result})
            if progress_callback is not None:
                progress_callback(current, total)
            if status_callback is not None:
                status_callback(current, total, "completed")
        return artifact_store.write_json(output_json_path, results)


def build_dspy_generator(llm: Any) -> Callable[[dict[str, Any]], Any]:
    """Create the complete planner/analyzer/renderer generator from dspy_prompt_test."""
    from feverslop.prompting.dspy_h3_generator import VideoPromptGenerator

    return VideoPromptGenerator(
        base_guide_path="minimax-h3-base.md",
        reference_guide_path="minimax-h3-references.md",
        llm=llm,
    )
