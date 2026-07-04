from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


def enrich_movie_render_plan_with_msr_prompts(*, project_dir: Path) -> Path:
    project_dir = Path(project_dir)
    movie_dir = project_dir / "movie"
    bible = _read_json(movie_dir / "bible.json")
    render_plan_path = movie_dir / "render_plan.json"
    reference_manifest_path = movie_dir / "references" / "manifest.json"
    continuity_plan_path = movie_dir / "continuity_plan.json"
    render_plan = _read_json(render_plan_path)
    manifest = _read_json(reference_manifest_path)
    continuity_plan = _read_json(continuity_plan_path) if continuity_plan_path.exists() else {}

    enriched = deepcopy(render_plan)
    enriched["movie_bible_path"] = "movie/bible.json"
    if continuity_plan:
        enriched["movie_continuity_plan_path"] = "movie/continuity_plan.json"
    enriched["reference_manifest_path"] = "movie/references/manifest.json"
    enriched["msr_enriched"] = True
    enriched["shots"] = [
        _enrich_shot(shot, bible=bible, manifest=manifest, continuity_plan=continuity_plan, fps=_fps(bible))
        for shot in render_plan.get("shots") or []
    ]

    output_path = movie_dir / "render_plan_msr.json"
    output_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def _enrich_shot(shot: dict, *, bible: dict, manifest: dict, continuity_plan: dict, fps: int) -> dict:
    enriched = deepcopy(shot)
    prompt = _movie_video_prompt(shot, bible=bible, manifest=manifest, continuity_plan=continuity_plan)
    frame_count = max(1, int(round(float(shot.get("duration_seconds") or 1) * max(1, fps))))
    enriched["ltx"] = {
        **dict(enriched.get("ltx") or {}),
        "original_style_i2v_prompt": prompt,
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
    return enriched


def _movie_video_prompt(shot: dict, *, bible: dict, manifest: dict, continuity_plan: dict) -> str:
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
        f"Dialogue language: {dialogue_language}. All spoken dialogue/native audio must be spoken in {dialogue_language} only" if dialogue_language else "",
        f"Continuity: {shot.get('continuity_notes')}" if shot.get("continuity_notes") else "",
        _continuity_contract(shot, continuity_plan),
        f"Style: {'; '.join(str(item) for item in bible.get('style_constraints') or [])}" if bible.get("style_constraints") else "",
    ]
    prompt = ". ".join(part.strip(" .") for part in parts if str(part).strip())
    return _strip_reference_sheet_language(prompt)


def _continuity_contract(shot: dict, continuity_plan: dict) -> str:
    if not continuity_plan:
        return ""
    shot_id = str(shot.get("shot_id") or "")
    packet = (continuity_plan.get("scene_continuity") or {}).get(shot_id) or {}
    narrative = _narrative_for_shot(shot_id, continuity_plan.get("narrative_chain") or [])
    lines = [
        "CONTINUITY CONTRACT:",
        _line("Incoming", packet.get("incoming")),
        _line("Must preserve", packet.get("required_carryovers")),
        _line("Allowed changes", packet.get("allowed_changes")),
        _line("Outgoing", packet.get("outgoing")),
        _line("Story before", narrative.get("story_state_before")),
        _line("Story after", narrative.get("story_state_after")),
        _line("Cause from previous", narrative.get("cause_from_previous")),
        _line("Narrative purpose", narrative.get("narrative_purpose")),
        _line("Conflict or tension", narrative.get("conflict_or_tension")),
        _line("Turning point", narrative.get("turning_point")),
        _line("Sets up next", narrative.get("sets_up_next")),
        "Preserve these continuity facts unless this shot explicitly changes them.",
    ]
    contract = "\n".join(line for line in lines if line)
    return contract if contract.strip() != "CONTINUITY CONTRACT:" else ""


def _narrative_for_shot(shot_id: str, chain: list) -> dict:
    for item in chain:
        if isinstance(item, dict) and str(item.get("shot_id") or "") == shot_id:
            return item
    return {}


def _line(label: str, value: object) -> str:
    if isinstance(value, list):
        text = "; ".join(str(item).strip() for item in value if str(item).strip())
    else:
        text = str(value or "").strip()
    return f"{label}: {text}" if text else ""


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
