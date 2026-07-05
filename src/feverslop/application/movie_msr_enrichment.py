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
        description = _describe_reference_item(actor)
        if description:
            parts.append(
                f"Reference image {index}: {description}. "
                f"Use reference image {index} for this subject's identity, face, body, wardrobe, and materials."
            )
    location_description = _describe_reference_item(location_item)
    if location_description:
        parts.append(
            f"Background reference: {location_description}. "
            "Use this image as the scene environment, lighting, color palette, atmosphere, and spatial setting."
        )
    return " ".join(parts).strip()


def _movie_video_prompt(shot: dict, *, bible: dict, manifest: dict) -> str:
    references = shot.get("reference_ids") or {}
    actor_ids = references.get("actors") or shot.get("actor_ids") or []
    location_id = references.get("location") or shot.get("location_id") or ""
    actor_names = _names_for_ids(manifest.get("actors") or bible.get("actors") or [], actor_ids)
    location_name = _name_for_id(manifest.get("locations") or bible.get("locations") or [], location_id)
    dialogue_language = str((bible.get("runtime_constraints") or {}).get("dialogue_language") or "").strip()
    parts = [
        str(shot.get("description") or "").strip(),
        f"Actors: {', '.join(actor_names)}" if actor_names else "",
        f"Location: {location_name}" if location_name else "",
        f"Action: {shot.get('action')}" if shot.get("action") else "",
        f"Camera: {shot.get('camera')}" if shot.get("camera") else "",
        f"Acting: {shot.get('acting') or shot.get('expression')}" if shot.get("acting") or shot.get("expression") else "",
        f"Dialogue for native audio: {shot.get('dialogue')}" if shot.get("dialogue") else "",
        _dialogue_audio_contract(shot),
        f"Dialogue language: {dialogue_language}. All spoken dialogue/native audio must be spoken in {dialogue_language} only" if dialogue_language else "",
        f"Style: {'; '.join(str(item) for item in bible.get('style_constraints') or [])}" if bible.get("style_constraints") else "",
    ]
    prompt = ". ".join(part.strip(" .") for part in parts if str(part).strip())
    return _strip_reference_sheet_language(prompt)


def _dialogue_audio_contract(shot: dict) -> str:
    dialogue = str(shot.get("dialogue") or "").strip()
    if dialogue:
        return f"Spoken dialogue contract: Only this exact scripted dialogue may be spoken: {dialogue}. Do not invent, repeat, paraphrase, or add other spoken lines."
    return "Spoken dialogue contract: No spoken dialogue in this shot. Do not invent spoken lines, narration, singing, chanting, murmuring words, or pseudo-dialogue."


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


def _describe_reference_item(item: dict) -> str:
    if not item:
        return ""
    name = str(item.get("name") or item.get("id") or "").strip(" .")
    role = str(item.get("role") or "").strip(" .")
    visual = str(item.get("visual_description") or "").strip(" .")
    image_prompt = str(item.get("image_prompt") or "").strip(" .")
    chunks = [chunk for chunk in (name, role, visual or image_prompt) if chunk]
    return ", ".join(chunks)


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
