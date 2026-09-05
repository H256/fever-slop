from __future__ import annotations

from copy import deepcopy
from typing import Any

from feverslop.domain.canonical_render_plan import PromptRole, stable_scene_id


def materialize_continuation_entries(scene: dict[str, Any], *, group: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand one semantic scene into addressable technical render entries."""
    segments = list(group.get("segments") or [])
    if not segments:
        raise ValueError("continuation group must contain technical segments")
    semantic_scene = int(scene["scene"])
    semantic_segment_id = str(
        scene.get("segment_id") or (scene.get("metadata") or {}).get("segment_id") or "",
    ).strip()
    if not semantic_segment_id:
        raise ValueError("continuation scene requires a semantic segment ID")

    entries: list[dict[str, Any]] = []
    for index, segment in enumerate(segments, start=1):
        technical_id = str(segment.get("segment_id") or "").strip()
        if not technical_id:
            raise ValueError("continuation segment requires a stable segment ID")
        entry = deepcopy(scene)
        entry["scene"] = semantic_scene * 1_000_000 + 1_000 + index
        entry["semantic_scene"] = semantic_scene
        entry["semantic_segment_id"] = semantic_segment_id
        entry["technical_segment_id"] = technical_id
        entry["continuation_group_id"] = str(group.get("group_id") or "").strip()
        entry["continuation_predecessor_id"] = (
            None if index == 1 else str(segments[index - 2]["segment_id"])
        )
        entry["segment_id"] = technical_id
        entry["abs_start_seconds"] = float(segment["start_seconds"])
        entry["abs_end_seconds"] = float(segment["end_seconds"])
        entry["duration_seconds"] = float(segment["duration_seconds"])
        entry["frame_count"] = round(entry["duration_seconds"] * int(entry["fps"]))
        entry["render_frame_count"] = int(segment.get("render_frame_count") or entry["frame_count"])
        entry["anchor_frames"] = int(segment.get("anchor_frames") or 0)
        entry["cut"] = index == 1
        metadata = dict(entry.get("metadata") or {})
        metadata.update({
            "segment_id": technical_id,
            "semantic_scene": semantic_scene,
            "continuation_group_id": entry["continuation_group_id"],
            "continuation_predecessor_id": entry["continuation_predecessor_id"],
        })
        entry["metadata"] = metadata
        canonical = entry.get("canonical")
        if isinstance(canonical, dict):
            canonical = deepcopy(canonical)
            canonical["segment_id"] = technical_id
            canonical["scene_id"] = stable_scene_id(technical_id)
            entry["canonical"] = canonical
        if index == 1:
            entry.setdefault("metadata", {})["continuation_groups"] = [deepcopy(group)]
        else:
            entry.setdefault("metadata", {}).pop("continuation_groups", None)
        _project_prompt_relay(entry, scene, segment)
        roles = (entry.get("canonical") or {}).get("roles") or {}
        relay_role = roles.get(str(PromptRole.LTX_RELAY)) or {}
        for owner in ("generated", "override"):
            owned = relay_role.get(owner)
            if isinstance(owned, dict) and isinstance(owned.get("value"), list):
                holder = {"fps": entry["fps"], "ltx": {"prompt_relay": owned["value"]}}
                _project_prompt_relay(holder, scene, segment)
                owned["value"] = holder["ltx"]["prompt_relay"]
        entries.append(entry)
    return entries


def project_continuation_sources(sources, generated):
    """Carry semantic overrides and reference bindings into technical identities."""
    groups = {}
    for entry in generated:
        for group in (entry.get("metadata") or {}).get("continuation_groups") or []:
            if entry.get("semantic_scene") is not None:
                groups[str(entry.get("semantic_segment_id") or "")] = group
    projected = []
    for source in sources:
        segment_id = str(
            source.get("segment_id") or (source.get("metadata") or {}).get("segment_id")
            or (source.get("canonical") or {}).get("segment_id") or "",
        )
        group = groups.get(segment_id)
        if group is not None and not source.get("technical_segment_id"):
            projected.extend(materialize_continuation_entries(source, group=group))
        else:
            projected.append(source)
    return projected


def _project_prompt_relay(entry: dict[str, Any], source: dict[str, Any], segment: dict[str, Any]) -> None:
    ltx = entry.get("ltx")
    if not isinstance(ltx, dict) or not isinstance(ltx.get("prompt_relay"), list):
        return
    fps = int(entry["fps"])
    offset = round((float(segment["start_seconds"]) - float(source["abs_start_seconds"])) * fps)
    length = round(float(segment["duration_seconds"]) * fps)
    projected = []
    for relay in ltx["prompt_relay"]:
        if not isinstance(relay, dict):
            continue
        start = max(0, int(relay.get("frame_start", 0)) - offset)
        end = min(length, int(relay.get("frame_end", 0)) - offset)
        if end > start:
            item = deepcopy(relay)
            item["frame_start"] = start
            item["frame_end"] = end
            projected.append(item)
    ltx["prompt_relay"] = projected or [{
        "frame_start": 0,
        "frame_end": length,
        "state": "instrumental",
        "prompt": str(ltx.get("base_prompt") or ""),
    }]
