from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

from feverslop.domain.vision_references import ReferenceImage
from feverslop.errors import FeverSlopLMLError
from feverslop.ports.llm import LLMPort, VisionLLMPort
from feverslop.prompting.guide_loader import load_markdown_guide
from feverslop.prompting.msr_modules import MSRPromptModules
from feverslop.utils.io import atomic_write_json

logger = logging.getLogger(__name__)


def enrich_render_plan_with_msr_prompts(
    render_plan_path: str | Path,
    output_path: str | Path,
    *,
    llm: VisionLLMPort | None = None,
    on_analysis_status: Callable[[int, list[dict[str, str]]], None] | None = None,
    on_scene_complete: Callable[[int, int, int], None] | None = None,
) -> Path:
    render_plan_path = Path(render_plan_path)
    output_path = Path(output_path)
    try:
        render_plan = json.loads(render_plan_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Render plan contains invalid JSON: {render_plan_path}\n{e}",
        ) from e

    enriched = []
    total = len(render_plan)
    for index, scene in enumerate(render_plan, start=1):
        enriched_scene = enrich_scene_with_msr_prompts(
            scene,
            llm=llm,
            project_base=_reference_base(render_plan_path, scene),
            on_analysis_status=on_analysis_status,
        )
        enriched.append(enriched_scene)
        if on_scene_complete is not None:
            on_scene_complete(int(scene.get("scene", index)), index, total)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, enriched)
    _validate_written_render_plan(output_path, enriched)
    return output_path


def _validate_written_render_plan(output_path: Path, expected: list[dict]) -> None:
    """Verify the persisted enrichment artifact before downstream consumers use it."""
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise OSError(f"Enriched render plan was not written: {output_path}")
    try:
        persisted = json.loads(output_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Enriched render plan contains invalid JSON: {output_path}") from exc
    if not isinstance(persisted, list) or len(persisted) != len(expected):
        raise ValueError(f"Enriched render plan has an unexpected scene structure: {output_path}")
    if [scene.get("scene") for scene in persisted] != [scene.get("scene") for scene in expected]:
        raise ValueError(f"Enriched render plan scene structure does not match input: {output_path}")
    if persisted != expected:
        raise ValueError(f"Enriched render plan content does not match written data: {output_path}")


def enrich_scene_with_msr_prompts(
    scene: dict,
    *,
    llm: VisionLLMPort | None = None,
    project_base: Path | None = None,
    on_analysis_status: Callable[[int, list[dict[str, str]]], None] | None = None,
) -> dict:
    result = deepcopy(scene)
    ltx = result.setdefault("ltx", {})
    references = result.get("references") or {}
    relays = _effective_relays(result, list(ltx.get("prompt_relay") or []))
    if _scene_silent_mode(result):
        ltx["prompt_relay"] = relays

    global_prompt = build_msr_global_prompt(references)
    ltx["msr_global_prompt"] = global_prompt
    ltx["msr_preroll_prompt"] = _build_preroll_prompt(result)
    ltx["msr_tail_prompt"] = _build_tail_prompt(result)
    if not relays:
        return result

    vision = _build_vision_msr_prompts(
        result,
        relays,
        llm=llm,
        project_base=project_base,
        on_analysis_status=on_analysis_status,
    )
    if vision is not None:
        global_prompt, llm_prompts = vision
    else:
        llm_prompts = _build_llm_segment_prompts(result, relays, llm=llm)
    ltx["msr_global_prompt"] = global_prompt
    msr_relays = []
    for index, relay in enumerate(relays):
        msr_relay = dict(relay)
        prompt = llm_prompts.get(index) or _fallback_segment_prompt(result, relay)
        msr_relay["prompt"] = _clean_segment_prompt(prompt)
        msr_relays.append(msr_relay)
    ltx["msr_prompt_relay"] = msr_relays
    return result


def _build_vision_msr_prompts(
    scene: dict,
    relays: list[dict],
    *,
    llm: VisionLLMPort | None,
    project_base: Path | None,
    on_analysis_status: Callable[[int, list[dict[str, str]]], None] | None,
) -> tuple[str, dict[int, str]] | None:
    references = _scene_reference_images(scene, project_base=project_base)
    scene_number = int(scene.get("scene", 0))
    if not references:
        logger.warning("MSR image analysis fallback: scene=%s reason=no images", scene_number)
        return None
    modules = _dspy_modules(llm)
    if modules is None:
        logger.warning("MSR image analysis fallback: scene=%s reason=vision unavailable", scene_number)
        return None

    status_references = [{"id": ref.id, "type": ref.type} for ref in references]
    logger.info(
        "MSR image analysis attempt: scene=%s reference_count=%s references=%s",
        scene_number,
        len(references),
        status_references,
    )
    if on_analysis_status is not None:
        on_analysis_status(scene_number, status_references)
    try:
        payload = _msr_segment_payload(scene, relays)
        typed = modules.vision(payload, [reference.path for reference in references])
        data = typed.model_dump()
    except (FeverSlopLMLError, ConnectionError, TimeoutError, OSError, RuntimeError) as exc:
        logger.warning(
            "MSR image analysis fallback: scene=%s reason=vision unavailable",
            scene_number,
            exc_info=exc,
        )
        return None
    except (ValueError, TypeError) as exc:
        logger.warning(
            "MSR image analysis fallback: scene=%s reason=invalid response",
            scene_number,
            exc_info=exc,
        )
        return None
    try:
        parsed_references = data.get("references")
        parsed_relays = data.get("relays")
        expected_pairs = {(reference.id, reference.type) for reference in references}
        if not isinstance(parsed_references, list) or not isinstance(parsed_relays, list):
            raise ValueError("missing lists")
        descriptions: dict[tuple[str, str], str] = {}
        for item in parsed_references:
            pair = (str(item.get("id") or ""), str(item.get("type") or ""))
            description = str(item.get("description") or "").strip()
            if not all(pair) or not description or pair in descriptions:
                raise ValueError("invalid reference")
            descriptions[pair] = description
        if set(descriptions) != expected_pairs:
            raise ValueError("reference mismatch")
        prompts: dict[int, str] = {}
        for item in parsed_relays:
            index = int(item["index"])
            if index in prompts or not 0 <= index < len(relays):
                raise ValueError("invalid relay index")
            prompt = _clean_segment_prompt(str(item.get("prompt") or ""))
            if not _is_valid_segment_prompt(prompt, relays[index]):
                raise ValueError("invalid relay prompt")
            prompts[index] = prompt
        if set(prompts) != set(range(len(relays))):
            raise ValueError("missing relay index")
    except (FeverSlopLMLError, ValueError, TypeError, KeyError, IndexError) as exc:
        logger.warning(
            "MSR image analysis fallback: scene=%s reason=invalid response",
            scene_number,
            exc_info=exc,
        )
        return None

    parts = []
    for index, reference in enumerate(references, start=1):
        suffix = " (scene)" if reference.type == "location" else ""
        parts.append(f"Reference image {index}{suffix}: {descriptions[(reference.id, reference.type)]}.")
    return "\n\n".join(parts), prompts


def _scene_reference_images(scene: dict, *, project_base: Path | None) -> list[ReferenceImage]:
    references = scene.get("references") or {}
    actors = references.get("actor_reference_descriptions") or []
    paths = list(references.get("actor_msr_paths") or references.get("actor_sheet_paths") or [])
    if len(paths) != len(actors):
        return []
    candidates = [(str(item.get("id") or ""), "actor", path) for item, path in zip(actors, paths)]
    location = references.get("location_reference_description") or {}
    location_path = references.get("location_msr_path") or references.get("location_sheet_path")
    if location and not location_path:
        return []
    if location:
        candidates.append((str(location.get("id") or ""), "location", location_path))
    result = []
    for reference_id, reference_type, raw_path in candidates:
        path = Path(str(raw_path))
        if not path.is_absolute() and project_base is not None:
            path = project_base / path
        if not reference_id or not path.is_file():
            return []
        result.append(ReferenceImage(reference_id, reference_type, path))
    return result


def _reference_base(render_plan_path: Path, scene: dict) -> Path:
    raw_paths = list((scene.get("references") or {}).get("actor_msr_paths") or [])
    raw_paths.append((scene.get("references") or {}).get("location_msr_path") or "")
    for parent in (render_plan_path.parent, *render_plan_path.parents):
        if any(raw and (parent / str(raw)).is_file() for raw in raw_paths):
            return parent
    return render_plan_path.parent


def _msr_vision_system_prompt() -> str:
    return load_markdown_guide("msr-vision")


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
    modules = _dspy_modules(llm)
    if modules is None:
        logger.warning("MSR segment prompt generation fallback: DSPy unavailable")
        return {}

    try:
        payload = _msr_segment_payload(scene, relays)
        items = [item.model_dump() for item in modules.segments(payload).relays]
    except (FeverSlopLMLError, ConnectionError, TimeoutError, OSError, RuntimeError, ValueError, TypeError) as exc:
        logger.warning("MSR segment prompt generation failed; using deterministic fallback", exc_info=exc)
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


def _dspy_modules(llm: Any) -> MSRPromptModules | None:
    if not isinstance(getattr(llm, "model", None), str) or getattr(llm, "client", None) is None:
        return None
    try:
        return MSRPromptModules(llm)
    except (ImportError, RuntimeError):
        return None


def _msr_segment_system_prompt() -> str:
    return load_markdown_guide("msr-segments")


def _msr_segment_payload(scene: dict, relays: list[dict]) -> dict:
    references = scene.get("references") or {}
    metadata = scene.get("metadata") or {}
    return {
        "scene": scene.get("scene"),
        "actors": references.get("actor_reference_descriptions") or [],
        "location": references.get("location_reference_description") or {},
        "scene_type": metadata.get("type", ""),
        "silent_mode": _scene_silent_mode(scene),
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
    state = _effective_state(scene, relay)
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
        f"Cinematic atmosphere holds around {location}; {atmosphere}; {motion}; {camera_text} before the main action begins.",
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
    if state == "dialogue":
        return ("speak" in lower or "talk" in lower or "say" in lower) and (
            "lip sync" in lower or "lip-sync" in lower
        )
    return "lip sync" not in lower and "lip-sync" not in lower


def _scene_silent_mode(scene: dict) -> bool:
    metadata = scene.get("metadata") or {}
    return bool(metadata.get("silent_mode") or scene.get("silent_mode"))


def _effective_state(scene: dict, relay: dict) -> str:
    if _scene_silent_mode(scene):
        return "instrumental"
    return str(relay.get("state") or "").strip().lower()


def _effective_relays(scene: dict, relays: list[dict]) -> list[dict]:
    if not _scene_silent_mode(scene):
        return relays
    normalized = []
    for relay in relays:
        item = dict(relay)
        item["state"] = "instrumental"
        item["lyrics"] = ""
        item["text"] = ""
        item["prompt"] = _fallback_segment_prompt(scene, item)
        normalized.append(item)
    return normalized


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    original = text
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end < start:
        if end == -1:
            candidate = text[start:].strip().rstrip(",") + "]"
        else:
            candidate = text[start:end + 1]
    else:
        candidate = text[start:end + 1]
    candidate = re.sub(r",\s*]", "]", candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        object_texts = re.findall(r"\{[^{}]*\}", candidate, flags=re.DOTALL)
        objects = []
        for obj_text in object_texts:
            try:
                objects.append(json.loads(obj_text))
            except json.JSONDecodeError:
                continue
        if objects:
            return objects
        raise ValueError(
            "Could not parse JSON array from LLM response.\n"
            f"Original response:\n{original}\n\n"
            f"Candidate:\n{candidate}",
        )
