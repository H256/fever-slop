from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable
import json
import re

from feverslop.ports.llm import LLMPort


def enrich_render_plan_with_msr_prompts(
    render_plan_path: str | Path,
    output_path: str | Path,
    *,
    llm: LLMPort | None = None,
    on_scene_complete: Callable[[int, int, int], None] | None = None,
) -> Path:
    render_plan_path = Path(render_plan_path)
    output_path = Path(output_path)
    render_plan = json.loads(render_plan_path.read_text(encoding="utf-8-sig"))

    enriched = []
    total = len(render_plan)
    for index, scene in enumerate(render_plan, start=1):
        enriched_scene = enrich_scene_with_msr_prompts(scene, llm=llm)
        enriched.append(enriched_scene)
        if on_scene_complete is not None:
            on_scene_complete(int(scene.get("scene", index)), index, total)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def enrich_scene_with_msr_prompts(scene: dict, *, llm: LLMPort | None = None) -> dict:
    result = deepcopy(scene)
    ltx = result.setdefault("ltx", {})
    references = result.get("references") or {}
    relays = list(ltx.get("prompt_relay") or [])

    ltx["msr_global_prompt"] = build_msr_global_prompt(references)
    ltx["msr_preroll_prompt"] = _build_preroll_prompt(result)
    ltx["msr_tail_prompt"] = _build_tail_prompt(result)
    if not relays:
        return result

    llm_prompts = _build_llm_segment_prompts(result, relays, llm=llm)
    msr_relays = []
    for index, relay in enumerate(relays):
        msr_relay = dict(relay)
        prompt = llm_prompts.get(index) or _fallback_segment_prompt(result, relay)
        msr_relay["prompt"] = _clean_segment_prompt(prompt)
        msr_relays.append(msr_relay)
    ltx["msr_prompt_relay"] = msr_relays
    return result


def build_msr_global_prompt(references: dict) -> str:
    parts: list[str] = []
    for index, actor in enumerate(references.get("actor_reference_descriptions") or [], start=1):
        actor_text = _describe_reference_item(actor)
        if actor_text:
            parts.append(f"Reference image {index}: {actor_text}.")

    location = references.get("location_reference_description") or {}
    location_text = _describe_reference_item(location)
    if location_text:
        parts.append(f"Reference image {len(parts) + 1} (scene): {location_text}.")

    return "\n\n".join(parts).strip()


def _build_llm_segment_prompts(scene: dict, relays: list[dict], *, llm: LLMPort | None) -> dict[int, str]:
    if llm is None:
        return {}

    response = llm.complete_prompt(
        system_prompt=_msr_segment_system_prompt(),
        prompt=json.dumps(_msr_segment_payload(scene, relays), ensure_ascii=False, indent=2),
    )
    try:
        items = _extract_json_array(response)
    except Exception:
        return {}

    prompts: dict[int, str] = {}
    for item in items:
        try:
            index = int(item["index"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= index < len(relays):
            prompt = _clean_segment_prompt(str(item.get("prompt", "")))
            if _is_valid_segment_prompt(prompt, relays[index]):
                prompts[index] = prompt
    return prompts


def _msr_segment_system_prompt() -> str:
    return """
You write LTX MSR PromptRelay local segment directions.

Return ONLY valid JSON array with exactly one object per relay segment:
[
  {"index": 0, "prompt": "specific stage direction"}
]

Rules:
- Write in English.
- Segment prompts are stage direction, not edit instructions.
- Do not write "preserve same subject", "keep identity", "keep same subject", "lock first frame", or "Start frame".
- Do not repeat the full reference image descriptions.
- Use the named reference actor as the subject anchor.
- For state "singing": the actor sings the provided lyrics with clear lip sync and expressive acting.
- For non-singing states: the actor is silent, mouth closed, and physically performs the scene action.
- Include camera motion and concrete character/environment motion when provided.
- Write rich cinematic direction, usually 25 to 45 words per segment, with action, acting, camera behavior, and visible environment effects.
- Preroll-like or transition-like segments still need concrete cinematic atmosphere and tension, not generic continuity filler.
- No markdown, no comments, no frame numbers.
""".strip()


def _msr_segment_payload(scene: dict, relays: list[dict]) -> dict:
    references = scene.get("references") or {}
    metadata = scene.get("metadata") or {}
    return {
        "scene": scene.get("scene"),
        "actors": references.get("actor_reference_descriptions") or [],
        "location": references.get("location_reference_description") or {},
        "scene_type": metadata.get("type", ""),
        "lyrics": metadata.get("lyrics", ""),
        "base_concept": metadata.get("base_concept", ""),
        "camera_motion": metadata.get("camera_motion", ""),
        "character_motion": metadata.get("character_motion", ""),
        "relay_segments": [
            {
                "index": index,
                "frame_start": int(relay.get("frame_start", 0)),
                "frame_end": int(relay.get("frame_end", 0)),
                "state": relay.get("state", ""),
                "current_prompt": relay.get("prompt", ""),
            }
            for index, relay in enumerate(relays)
        ],
    }


def _fallback_segment_prompt(scene: dict, relay: dict) -> str:
    metadata = scene.get("metadata") or {}
    actor = _primary_actor_name(scene.get("references") or {})
    location = _location_name(scene.get("references") or {})
    state = str(relay.get("state") or "").strip().lower()
    lyrics = str(metadata.get("lyrics") or "").strip().strip(".")
    camera = str(metadata.get("camera_motion") or "").strip()
    character_motion = str(metadata.get("character_motion") or "").strip()
    base_concept = str(metadata.get("base_concept") or "").strip()

    if state == "singing":
        lyric_phrase = f' the phrase "{lyrics}"' if lyrics else ""
        action = f"{actor} sings{lyric_phrase} with clear lip sync and expressive acting"
    else:
        action = f"{actor} stays silent with mouth closed"

    motion = character_motion or base_concept or "the scene action builds with controlled physical intensity"
    camera_text = camera or "the camera holds a readable cinematic view"
    environment = base_concept or f"the atmosphere of {location} remains visible around the reference subject"
    return _clean_segment_prompt(f"{action}; {motion}; {camera_text}; {environment}.")


def _build_preroll_prompt(scene: dict) -> str:
    metadata = scene.get("metadata") or {}
    references = scene.get("references") or {}
    actor = _primary_actor_name(references)
    location = _location_name(references)
    base_concept = str(metadata.get("base_concept") or "").strip()
    camera = str(metadata.get("camera_motion") or "").strip()
    character_motion = str(metadata.get("character_motion") or "").strip()
    atmosphere = base_concept or f"atmospheric detail gathers across {location}"
    motion = character_motion or f"{actor} remains physically present as the tension builds"
    camera_text = camera or "the camera holds a steady cinematic setup"
    return _clean_segment_prompt(
        f"Cinematic atmosphere holds around {location}; {atmosphere}; {motion}; {camera_text} before the main action begins."
    )


def _build_tail_prompt(scene: dict) -> str:
    metadata = scene.get("metadata") or {}
    references = scene.get("references") or {}
    actor = _primary_actor_name(references)
    location = _location_name(references)
    base_concept = str(metadata.get("base_concept") or "").strip()
    camera = str(metadata.get("camera_motion") or "").strip()
    character_motion = str(metadata.get("character_motion") or "").strip()
    motion = character_motion or f"{actor} carries the last action forward"
    environment = base_concept or f"the atmosphere of {location} keeps reacting around the subject"
    camera_text = camera or "the camera continues the same cinematic movement"
    return _clean_segment_prompt(f"{motion} through {location}; {environment}; {camera_text}; the energy resolves without a new scene.")


def _primary_actor_name(references: dict) -> str:
    actors = references.get("actor_reference_descriptions") or []
    if actors:
        name = str(actors[0].get("name") or actors[0].get("id") or "").strip()
        if name:
            return name
    return "The reference actor"


def _location_name(references: dict) -> str:
    location = references.get("location_reference_description") or {}
    name = str(location.get("name") or location.get("id") or "").strip()
    return name or "the referenced location"


def _describe_reference_item(item: dict) -> str:
    name = str(item.get("name") or item.get("id") or "").strip(" .")
    role = str(item.get("role") or "").strip(" .")
    visual = str(item.get("visual_description") or "").strip(" .")
    image_prompt = str(item.get("image_prompt") or "").strip(" .")
    return ", ".join(chunk for chunk in (name, role, visual or image_prompt) if chunk)


def _clean_segment_prompt(prompt: str) -> str:
    cleaned = " ".join(str(prompt or "").replace("\n", " ").split())
    cleaned = re.sub(r"(?is)\bStart frame:\s*", "", cleaned)
    cleaned = re.sub(r"(?is)\bLock the first frame\b.*?(?:\.|$)", "", cleaned)
    cleaned = re.sub(r"(?is)\bpreserve same shot\b", "", cleaned)
    cleaned = re.sub(r"(?is)\bpreserve the same shot\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;")
    return cleaned


def _is_valid_segment_prompt(prompt: str, relay: dict) -> bool:
    lower = prompt.lower()
    banned = (
        "preserve same subject",
        "keep identity",
        "keep same subject",
        "lock first frame",
        "start frame",
    )
    if not prompt or any(text in lower for text in banned):
        return False
    state = str(relay.get("state") or "").strip().lower()
    if state == "singing":
        return "sing" in lower and ("lip sync" in lower or "lip-sync" in lower)
    return "lip sync" not in lower and "lip-sync" not in lower


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    original = text
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON array found in LLM response:\n{original}")
    return json.loads(text[start:end + 1])
