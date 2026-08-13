from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from feverslop.application.reference_bible import (
    build_runtime_consistency_contract,
    visual_consistency_sources,
)
from feverslop.domain.visual_consistency_runtime import (
    bind_continuity_anchors,
    reference_look_id,
    resolve_reference_look,
)
from feverslop.application.msr_prompt_enrichment import _clean_segment_prompt, _is_valid_segment_prompt, _msr_vision_system_prompt
from feverslop.domain.llm_parsing import extract_json_object
from feverslop.domain.screenplay import looks_like_screenplay
from feverslop.domain.vision_references import ReferenceImage
from feverslop.ports.llm import VisionLLMPort
from feverslop.utils.io import atomic_write_json, read_json_object

logger = logging.getLogger(__name__)


def enrich_movie_render_plan_with_msr_prompts(
    *,
    project_dir: Path,
    keyframe_mode: str = "none",
    llm: VisionLLMPort | None = None,
    on_analysis_status=None,
    workflow_profile: str = "msr-default",
) -> Path:
    project_dir = Path(project_dir)
    movie_dir = project_dir / "movie"
    bible = _read_json(movie_dir / "bible.json")
    render_plan_path = movie_dir / "render_plan.json"
    reference_manifest_path = movie_dir / "references" / "manifest.json"
    continuity_plan_path = movie_dir / "continuity_plan.json"
    shot_cards_path = movie_dir / "shot_cards.json"
    render_plan = _read_json(render_plan_path)
    manifest = _read_json(reference_manifest_path)
    try:
        continuity_plan = _read_json(continuity_plan_path)
    except (FileNotFoundError, IsADirectoryError):
        continuity_plan = {}
    try:
        shot_cards = _read_json(shot_cards_path)
    except (FileNotFoundError, IsADirectoryError):
        shot_cards = {}

    enriched = deepcopy(render_plan)
    enriched["movie_bible_path"] = "movie/bible.json"
    if continuity_plan:
        enriched["movie_continuity_plan_path"] = "movie/continuity_plan.json"
    enriched["reference_manifest_path"] = "movie/references/manifest.json"
    if shot_cards:
        enriched["movie_shot_cards_path"] = "movie/shot_cards.json"
    enriched["keyframe_mode"] = keyframe_mode
    enriched["msr_enriched"] = True
    enriched["shots"] = [
        _enrich_shot(
            {**shot, "scene": int(shot.get("scene") or index)},
            bible=bible, manifest=manifest, fps=_fps(bible), keyframe_mode=keyframe_mode,
            shot_cards=shot_cards, project_dir=project_dir, llm=llm, on_analysis_status=on_analysis_status,
            workflow_profile=workflow_profile,
        )
        for index, shot in enumerate(render_plan.get("shots") or [], start=1)
    ]

    output_path = movie_dir / "render_plan_msr.json"
    atomic_write_json(output_path, enriched)
    return output_path


def _enrich_shot(
    shot: dict, *, bible: dict, manifest: dict, fps: int, keyframe_mode: str = "none",
    shot_cards: dict | None = None, project_dir: Path | None = None, llm: VisionLLMPort | None = None,
    on_analysis_status=None,
    workflow_profile: str = "msr-default",
) -> dict:
    enriched = deepcopy(shot)
    shot_card = _shot_card_for_id(shot_cards or {}, str(shot.get("shot_id") or ""))
    prompt = _movie_video_prompt(shot, bible=bible, manifest=manifest)
    continuity_notes = "; ".join(_safe_continuity_facts(shot.get("continuity_notes")))
    if continuity_notes:
        enriched["continuity_notes"] = continuity_notes
    elif "continuity_notes" in enriched:
        enriched["continuity_notes"] = ""
    frame_count = max(1, int(round(float(shot.get("duration_seconds") or 1) * max(1, fps))))
    fallback_global = _movie_reference_global_prompt(shot, bible=bible, manifest=manifest)
    contract, contract_sources = _movie_runtime_contract(
        shot,
        bible=bible,
        manifest=manifest,
        project_dir=project_dir,
        workflow_profile=workflow_profile,
    )
    vision = _movie_vision_prompts(
        shot, bible=bible, manifest=manifest, project_dir=project_dir, llm=llm,
        frame_count=frame_count, on_analysis_status=on_analysis_status,
    )
    global_prompt, relay_prompt = vision or (fallback_global, prompt)
    global_prompt = bind_continuity_anchors(global_prompt, contract)
    relay_prompt = bind_continuity_anchors(relay_prompt, None)
    if contract is not None:
        enriched["visual_consistency"] = contract.to_dict()
        enriched["visual_consistency_sources"] = contract_sources
    enriched["ltx"] = {
        **dict(enriched.get("ltx") or {}),
        "original_style_i2v_prompt": prompt,
        "msr_global_prompt": global_prompt,
        "native_audio": True,
        "msr_prompt_relay_mode": "single",
        "msr_prompt_relay": [
            {
                "frame_start": 0,
                "frame_end": frame_count - 1,
                "prompt": relay_prompt,
                "camera": str(shot.get("camera") or "").strip(),
                "acting": str(shot.get("acting") or shot.get("expression") or "").strip(),
                "dialogue": str(shot.get("dialogue") or "").strip(),
            }
        ],
    }
    if keyframe_mode in {"start", "start-end"}:
        keyframes = dict(enriched.get("keyframes") or {})
        if shot_card.get("start_frame_brief") and not keyframes.get("start_frame_prompt"):
            keyframes["start_frame_prompt"] = shot_card["start_frame_brief"]
        if keyframe_mode == "start-end" and shot_card.get("end_frame_brief") and not keyframes.get("end_frame_prompt"):
            keyframes["end_frame_prompt"] = shot_card["end_frame_brief"]
        enriched["keyframes"] = keyframes
    return enriched


def _movie_runtime_contract(
    shot: dict,
    *,
    bible: dict,
    manifest: dict,
    project_dir: Path | None,
    workflow_profile: str,
):
    if project_dir is None or type(shot.get("scene")) is not int:
        return None, {}
    reference_images = _movie_reference_images(
        shot,
        bible=bible,
        manifest=manifest,
        project_dir=project_dir,
    )
    metadata = _movie_reference_metadata(shot, bible=bible, manifest=manifest)
    if not reference_images or len(reference_images) != len(metadata):
        return None, {}
    images = [
        {
            **item,
            "id": reference.id,
            "type": reference.type,
            "path": _project_relative_reference(
                reference,
                project_dir=project_dir,
            ),
        }
        for item, reference in zip(metadata, reference_images)
    ]
    return (
        build_runtime_consistency_contract(
            shot,
            images=images,
            project_base=project_dir,
            mode="msr",
            workflow_profile=workflow_profile,
        ),
        visual_consistency_sources(images, project_base=project_dir),
    )


def _project_relative_reference(
    reference: ReferenceImage,
    *,
    project_dir: Path,
) -> str:
    root = Path(project_dir).resolve()
    path = reference.path.resolve()
    if not path.is_relative_to(root):
        raise ValueError(
            f"Movie MSR reference {reference.id!r} path must be "
            f"project-relative and inside the project: {reference.path}"
        )
    return path.relative_to(root).as_posix()


def _movie_vision_prompts(
    shot: dict, *, bible: dict, manifest: dict, project_dir: Path | None, llm: VisionLLMPort | None,
    frame_count: int, on_analysis_status,
) -> tuple[str, str] | None:
    shot_id = str(shot.get("shot_id") or "")
    references = _movie_reference_images(shot, bible=bible, manifest=manifest, project_dir=project_dir)
    if not references:
        logger.warning("MSR image analysis fallback: shot=%s reason=no images", shot_id)
        return None
    complete_with_images = getattr(llm, "complete_prompt_with_images", None)
    if not callable(complete_with_images):
        logger.warning("MSR image analysis fallback: shot=%s reason=vision unavailable", shot_id)
        return None
    status_references = [{"id": ref.id, "type": ref.type} for ref in references]
    logger.info(
        "MSR image analysis attempt: shot=%s reference_count=%s references=%s",
        shot_id, len(references), status_references,
    )
    if on_analysis_status is not None:
        on_analysis_status(shot_id, status_references)
    relay = {"frame_start": 0, "frame_end": frame_count - 1, "state": _movie_relay_state(shot)}
    metadata = _movie_reference_metadata(shot, bible=bible, manifest=manifest)
    try:
        response = complete_with_images(
            _msr_vision_system_prompt(),
            json.dumps({"references": metadata, "shot_context": shot, "relay_segments": [{"index": 0, **relay}]}, ensure_ascii=True),
            [reference.path for reference in references],
        )
    except Exception:
        logger.warning("MSR image analysis fallback: shot=%s reason=vision unavailable", shot_id)
        return None
    try:
        data = extract_json_object(response)
        items = data.get("references")
        relays = data.get("relays")
        if not isinstance(items, list) or not isinstance(relays, list) or len(relays) != 1:
            raise ValueError("missing lists")
        descriptions = {}
        for item in items:
            pair = (str(item.get("id") or ""), str(item.get("type") or ""))
            description = str(item.get("description") or "").strip()
            if not all(pair) or not description or pair in descriptions:
                raise ValueError("invalid reference")
            descriptions[pair] = description
        if set(descriptions) != {(ref.id, ref.type) for ref in references}:
            raise ValueError("reference mismatch")
        if int(relays[0].get("index", -1)) != 0:
            raise ValueError("relay mismatch")
        relay_prompt = _clean_segment_prompt(str(relays[0].get("prompt") or ""))
        if not _is_valid_segment_prompt(relay_prompt, relay):
            raise ValueError("invalid relay")
    except Exception:
        logger.warning("MSR image analysis fallback: shot=%s reason=invalid response", shot_id)
        return None

    parts = []
    for index, reference in enumerate(references, start=1):
        item = metadata[index - 1]
        label = "Scene" if reference.type == "location" else str(item.get("name") or reference.id)
        parts.append(f"Reference image {index} ({label}): {descriptions[(reference.id, reference.type)]}.")
    return " ".join(parts), relay_prompt


def _movie_relay_state(shot: dict) -> str:
    if str(shot.get("dialogue") or "").strip():
        return "dialogue"
    mode = str(
        shot.get("performance_mode") or shot.get("state") or shot.get("type") or ""
    ).strip().lower()
    if mode in {"singing", "vocals", "vocal"} or str(shot.get("lyrics") or "").strip():
        return "singing"
    return "instrumental"


def _movie_reference_metadata(shot: dict, *, bible: dict, manifest: dict) -> list[dict]:
    ids = shot.get("reference_ids") or {}
    actor_ids = ids.get("actors") or shot.get("actor_ids") or []
    actor_items = manifest.get("actors") or bible.get("actors") or []
    actors = []
    for actor_id in actor_ids:
        item = _item_for_id(actor_items, actor_id)
        if item:
            actors.append(resolve_reference_look(
                item,
                reference_look_id(
                    shot,
                    kind="actor",
                    semantic_id=str(actor_id),
                ),
            ))
    location_id = ids.get("location") or shot.get("location_id") or ""
    location = _item_for_id(
        manifest.get("locations") or bible.get("locations") or [],
        location_id,
    )
    if location:
        location = resolve_reference_look(
            location,
            reference_look_id(
                shot,
                kind="location",
                semantic_id=str(location_id),
            ),
        )
    return [dict(item, type="actor") for item in actors] + ([dict(location, type="location")] if location else [])


def _movie_reference_images(shot: dict, *, bible: dict, manifest: dict, project_dir: Path | None) -> list[ReferenceImage]:
    requested = shot.get("reference_ids") or {}
    expected_pairs = [
        (str(actor_id), "actor")
        for actor_id in (requested.get("actors") or shot.get("actor_ids") or [])
    ]
    location_id = str(requested.get("location") or shot.get("location_id") or "")
    if location_id:
        expected_pairs.append((location_id, "location"))
    metadata = _movie_reference_metadata(shot, bible=bible, manifest=manifest)
    if [(str(item.get("id") or ""), str(item.get("type") or "")) for item in metadata] != expected_pairs:
        return []
    result = []
    for item in metadata:
        raw_path = str(
            (
                item.get("sheet_path")
                if item.get("look_id") != "default"
                else item.get("msr_sheet_path") or item.get("sheet_path")
            )
            or ""
        ).strip()
        path = Path(raw_path)
        if raw_path and not path.is_absolute() and project_dir is not None:
            path = project_dir / path
        if not item.get("id") or not path.is_file():
            return []
        result.append(ReferenceImage(str(item["id"]), str(item["type"]), path))
    return result


def _movie_reference_global_prompt(shot: dict, *, bible: dict, manifest: dict) -> str:
    references = shot.get("reference_ids") or {}
    actor_ids = references.get("actors") or shot.get("actor_ids") or []
    location_id = references.get("location") or shot.get("location_id") or ""
    actor_items = _items_for_ids(manifest.get("actors") or bible.get("actors") or [], actor_ids)
    location_item = _item_for_id(manifest.get("locations") or bible.get("locations") or [], location_id)
    parts = []
    for index, actor in enumerate(actor_items, start=1):
        description = _describe_reference_item(actor, item_type="actor")
        if description:
            label = _reference_label(actor, fallback=f"Actor {index}")
            parts.append(f"Reference image {index} ({label}): {description}.")
    location_description = _describe_reference_item(location_item, item_type="scene")
    if location_description:
        parts.append(f"Reference image {len(actor_items) + 1} (Scene): {location_description}.")
    return " ".join(parts).strip()


def _movie_video_prompt(shot: dict, *, bible: dict, manifest: dict) -> str:
    references = shot.get("reference_ids") or {}
    actor_ids = references.get("actors") or shot.get("actor_ids") or []
    actor_names = _names_for_ids(manifest.get("actors") or bible.get("actors") or [], actor_ids)
    cast_members = [
        f"{name} (`{actor_id}`)"
        for actor_id, name in zip(actor_ids, actor_names)
    ]
    cast_binding = f"Visible cast: {_natural_join(cast_members)}" if cast_members else ""
    description = _clean_movie_prompt_field(shot.get("description"))
    action = _clean_movie_prompt_field(shot.get("action"))
    camera = _clean_movie_prompt_field(shot.get("camera"))
    acting = _clean_movie_prompt_field(shot.get("acting") or shot.get("expression"))
    dialogue_language = str((bible.get("runtime_constraints") or {}).get("dialogue_language") or "").strip()
    dialogue_direction = _local_dialogue_direction(shot, actor_names, dialogue_language=dialogue_language)
    action_part = "" if action and dialogue_direction.casefold().startswith(action.casefold()) else action
    parts = [
        cast_binding,
        description,
        action_part if action_part and action_part != description else "",
        f"Camera: {camera}" if camera else "",
        f"Acting: {acting}" if acting else "",
        dialogue_direction,
    ]
    prompt = ". ".join(part.strip() for part in parts if str(part).strip())
    return _strip_reference_sheet_language(prompt)


def _natural_join(items: list[str]) -> str:
    if len(items) < 2:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _local_dialogue_direction(shot: dict, actor_names: list[str], *, dialogue_language: str) -> str:
    dialogue = str(shot.get("dialogue") or "").strip()
    if not dialogue:
        return ""
    cue, spoken_text = _dialogue_cue_and_text(dialogue)
    device = _diegetic_audio_device(cue, dialogue)
    language_phrase = f" in {dialogue_language}" if dialogue_language else ""
    if device:
        return _diegetic_audio_direction(device=device, spoken_text=spoken_text, shot=shot, language_phrase=language_phrase, actor_names=actor_names)
    speaker = _dialogue_speaker(dialogue, actor_names)
    verb = _dialogue_verb(spoken_text)
    if speaker:
        return f"{speaker} {verb}{language_phrase}: \"{spoken_text}\""
    fallback = actor_names[0] if actor_names else "The visible referenced actor"
    return f"{fallback} {verb}{language_phrase}: \"{spoken_text}\""


def _dialogue_cue_and_text(dialogue: str) -> tuple[str, str]:
    text = str(dialogue or "").strip()
    parenthetical = re.match(r"^\(([^)]+)\)\s*(.+)$", text, flags=re.DOTALL)
    if parenthetical:
        return parenthetical.group(1).strip(), parenthetical.group(2).strip()
    if ":" in text:
        cue, spoken = text.split(":", 1)
        return cue.strip(), spoken.strip()
    return "", _spoken_dialogue_text(text)


def _diegetic_audio_device(cue: str, dialogue: str) -> str:
    text = f"{cue} {dialogue}".casefold()
    if any(token in text for token in ("radio", "transmitter", "speaker", "recording", "distorted voice", "voice")):
        if "radio" in text or "transmitter" in text:
            return "radio"
        if "speaker" in text:
            return "speaker"
        return "radio"
    return ""


def _diegetic_audio_direction(*, device: str, spoken_text: str, shot: dict, language_phrase: str, actor_names: list[str]) -> str:
    action = _clean_movie_prompt_field(shot.get("action"))
    if action and ("voice" in action.casefold() or device in action.casefold()):
        phrase = action.rstrip(".")
        return f"{phrase}{language_phrase}: \"{spoken_text}\""
    if device == "speaker":
        return f"The speaker emits a diegetic voice{language_phrase}: \"{spoken_text}\""
    actor_name = actor_names[0] if actor_names else "the visible actor"
    return f"The radio plays a recording of {actor_name}'s own voice screaming{language_phrase}: \"{spoken_text}\""


def _dialogue_verb(spoken_text: str) -> str:
    return "asks" if str(spoken_text or "").strip().endswith("?") else "says"


def _dialogue_speaker(dialogue: str, actor_names: list[str]) -> str:
    cue = str(dialogue or "").split(":", 1)[0].strip()
    if ":" in str(dialogue or "") and cue:
        for actor_name in actor_names:
            if cue.casefold() == str(actor_name).strip().casefold():
                return str(actor_name).strip()
        return cue
    if len(actor_names) == 1:
        return actor_names[0]
    return ""


def _spoken_dialogue_text(dialogue: str) -> str:
    lines = []
    for raw_line in str(dialogue or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" in line:
            line = line.split(":", 1)[1].strip()
        if line:
            lines.append(line)
    return " ".join(lines).strip()


def _clean_movie_prompt_field(value: Any) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return ""
    text = re.split(
        r"\s+(?:story_idea|steering|prompt_guidance|dialogue_language)\s*:",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return text.strip(" .")


def _safe_continuity_facts(value: Any) -> tuple[str, ...]:
    if isinstance(value, list | tuple):
        candidates = [part for item in value for part in _split_continuity_text(item)]
    else:
        candidates = _split_continuity_text(value)
    facts: list[str] = []
    for candidate in candidates:
        fact = " ".join(str(candidate or "").split()).strip(" .")
        if not fact or _looks_like_screenplay_dump(fact):
            continue
        if fact not in facts:
            facts.append(fact)
    return tuple(facts)


def _split_continuity_text(value: Any) -> list[str]:
    text = str(value or "")
    return [part.strip() for part in re.split(r"[;\n]+", text) if part.strip()]


_looks_like_screenplay_dump = looks_like_screenplay


def _strip_reference_sheet_language(value: str) -> str:
    banned = (
        "Full-body cinematic character reference sheet",
        "Four vertical panels",
        "plain white seamless studio background",
        "no extra characters",
    )
    text = value
    for phrase in banned:
        text = text.replace(phrase, "")
    return " ".join(text.split()).strip()


def _names_for_ids(items: list[dict], ids: list[str]) -> list[str]:
    names = []
    by_id = {str(item.get("id")): str(item.get("name") or item.get("id")) for item in items if isinstance(item, dict)}
    for item_id in ids:
        name = by_id.get(str(item_id))
        if name and name not in names:
            names.append(name)
    return names


def _name_for_id(items: list[dict], item_id: str) -> str:
    for item in items:
        if isinstance(item, dict) and str(item.get("id")) == str(item_id):
            return str(item.get("name") or item_id)
    return str(item_id or "")


def _items_for_ids(items: list[dict], ids: list[str]) -> list[dict]:
    by_id = {str(item.get("id")): item for item in items if isinstance(item, dict)}
    return [by_id[str(item_id)] for item_id in ids if str(item_id) in by_id]


def _item_for_id(items: list[dict], item_id: str) -> dict:
    for item in items:
        if isinstance(item, dict) and str(item.get("id")) == str(item_id):
            return item
    return {}


def _reference_label(item: dict, *, fallback: str) -> str:
    return str(item.get("name") or item.get("id") or fallback).strip(" .") or fallback


def _describe_reference_item(item: dict, *, item_type: str) -> str:
    if not item:
        return ""
    name = str(item.get("name") or item.get("id") or "").strip(" .")
    visual = str(item.get("visual_description") or "").strip(" .")
    image_prompt = str(item.get("image_prompt") or "").strip(" .")
    text = visual or image_prompt or name
    return _clean_reference_description(text, name=name, item_type=item_type)


def _clean_reference_description(value: str, *, name: str, item_type: str) -> str:
    text = " ".join(str(value or "").split()).strip(" .")
    if not text:
        return name
    text = _strip_reference_sheet_language(text)
    prefixes = [name, "character", "scene", "location", "background"]
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            prefix = str(prefix or "").strip()
            if not prefix:
                continue
            pattern = rf"^{re.escape(prefix)}\s*,\s*"
            new_text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip(" .")
            if new_text != text:
                text = new_text
                changed = True
    if item_type == "actor":
        text = re.sub(r"\bstory-defined cinematic character with\b", "", text, flags=re.IGNORECASE)
    else:
        text = re.sub(r"\bstory-defined cinematic location with\b", "", text, flags=re.IGNORECASE)
    text = " ".join(text.split()).strip(" .,")
    generic_placeholders = {
        "consistent face, hair, body shape, wardrobe, and posture",
        "consistent production design, geography, lighting, and atmosphere",
    }
    if not text or text in generic_placeholders:
        return name
    return text


def _shot_card_for_id(shot_cards: dict, shot_id: str) -> dict:
    for card in shot_cards.get("shot_cards") or []:
        if isinstance(card, dict) and str(card.get("shot_id") or "") == shot_id:
            return card
    return {}


def _fps(bible: dict) -> int:
    runtime = bible.get("runtime_constraints") or {}
    return int(runtime.get("fps") or 24)


_read_json = read_json_object
