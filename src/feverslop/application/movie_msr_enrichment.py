from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path


_SCREENPLAY_HEADING_RE = re.compile(r"\b(?:INT|EXT|INT/EXT)\.\s+", re.IGNORECASE)
_DIALOGUE_CUE_RE = re.compile(r"\b[A-Z][A-Z0-9 _'-]{1,30}:\s+\S")


def enrich_movie_render_plan_with_msr_prompts(*, project_dir: Path, keyframe_mode: str = "none") -> Path:
    project_dir = Path(project_dir)
    movie_dir = project_dir / "movie"
    bible = _read_json(movie_dir / "bible.json")
    render_plan_path = movie_dir / "render_plan.json"
    reference_manifest_path = movie_dir / "references" / "manifest.json"
    continuity_plan_path = movie_dir / "continuity_plan.json"
    shot_cards_path = movie_dir / "shot_cards.json"
    render_plan = _read_json(render_plan_path)
    manifest = _read_json(reference_manifest_path)
    continuity_plan = _read_json(continuity_plan_path) if continuity_plan_path.exists() else {}
    shot_cards = _read_json(shot_cards_path) if shot_cards_path.exists() else {}

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
        _enrich_shot(shot, bible=bible, manifest=manifest, fps=_fps(bible), keyframe_mode=keyframe_mode, shot_cards=shot_cards)
        for shot in render_plan.get("shots") or []
    ]

    output_path = movie_dir / "render_plan_msr.json"
    output_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def _enrich_shot(shot: dict, *, bible: dict, manifest: dict, fps: int, keyframe_mode: str = "none", shot_cards: dict | None = None) -> dict:
    enriched = deepcopy(shot)
    shot_card = _shot_card_for_id(shot_cards or {}, str(shot.get("shot_id") or ""))
    prompt = _movie_video_prompt(shot, bible=bible, manifest=manifest)
    continuity_notes = "; ".join(_safe_continuity_facts(shot.get("continuity_notes")))
    if continuity_notes:
        enriched["continuity_notes"] = continuity_notes
    elif "continuity_notes" in enriched:
        enriched["continuity_notes"] = ""
    frame_count = max(1, int(round(float(shot.get("duration_seconds") or 1) * max(1, fps))))
    enriched["ltx"] = {
        **dict(enriched.get("ltx") or {}),
        "original_style_i2v_prompt": prompt,
        "msr_global_prompt": _movie_reference_global_prompt(shot, bible=bible, manifest=manifest),
        "native_audio": True,
        "msr_prompt_relay_mode": "single",
        "msr_prompt_relay": [
            {
                "frame_start": 0,
                "frame_end": frame_count - 1,
                "prompt": prompt,
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
    description = _clean_movie_prompt_field(shot.get("description"))
    action = _clean_movie_prompt_field(shot.get("action"))
    camera = _clean_movie_prompt_field(shot.get("camera"))
    acting = _clean_movie_prompt_field(shot.get("acting") or shot.get("expression"))
    dialogue_language = str((bible.get("runtime_constraints") or {}).get("dialogue_language") or "").strip()
    dialogue_direction = _local_dialogue_direction(shot, actor_names, dialogue_language=dialogue_language)
    action_part = "" if action and dialogue_direction.casefold().startswith(action.casefold()) else action
    parts = [
        description,
        action_part if action_part and action_part != description else "",
        f"Camera: {camera}" if camera else "",
        f"Acting: {acting}" if acting else "",
        dialogue_direction,
    ]
    prompt = ". ".join(part.strip() for part in parts if str(part).strip())
    return _strip_reference_sheet_language(prompt)


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
    return f"The visible referenced actor {verb}{language_phrase}: \"{spoken_text}\""


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


def _clean_movie_prompt_field(value: object) -> str:
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


def _safe_continuity_facts(value: object) -> tuple[str, ...]:
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


def _split_continuity_text(value: object) -> list[str]:
    text = str(value or "")
    return [part.strip() for part in re.split(r"[;\n]+", text) if part.strip()]


def _looks_like_screenplay_dump(text: str) -> bool:
    if len(text) > 300:
        return True
    if _SCREENPLAY_HEADING_RE.search(text):
        return True
    return bool(_DIALOGUE_CUE_RE.search(text))


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


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Movie pipeline artifact not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Movie pipeline artifact must be a JSON object: {path}")
    return data
