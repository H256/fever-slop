from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from feverslop.path_utils import coerce_local_path


def scene_duration_from_frame_count(frame_count: int, fps: int) -> float:
    return max(0.0, (int(frame_count) - 1) / float(fps))


def frame_count_from_duration(duration_seconds: float, fps: int) -> int:
    return round(float(duration_seconds) * int(fps)) + 1


def normalize_render_plan(
    render_plan: list[dict],
    min_duration: float,
    max_duration: float,
    renumber: bool = True,
) -> list[dict]:
    """Repair render_plan durations using the same min/max scene_generation criteria.

    This is a safety net. The primary fix should happen earlier on scene_srt.
    """
    if not render_plan:
        return []

    if min_duration <= 0:
        raise ValueError("min_duration must be > 0")

    if max_duration < min_duration:
        raise ValueError("max_duration must be >= min_duration")

    fps = int(render_plan[0]["fps"])
    min_frames = frame_count_from_duration(min_duration, fps)
    max_frames = frame_count_from_duration(max_duration, fps)

    groups: list[list[dict]] = []
    current: list[dict] = []

    for scene in render_plan:
        current.append(scene)

        start = float(current[0]["abs_start_seconds"])
        end = float(current[-1]["abs_end_seconds"])
        frame_count = frame_count_from_duration(end - start, fps)

        if frame_count >= min_frames:
            groups.append(current)
            current = []

    if current:
        if groups:
            groups[-1].extend(current)
        else:
            groups.append(current)

    normalized: list[dict] = []

    for group in groups:
        merged = _merge_group(group, fps=fps)

        if int(merged["frame_count"]) > max_frames:
            normalized.extend(_split_render_scene_to_max(merged, fps=fps, max_duration=max_duration))
        else:
            normalized.append(merged)

    if renumber:
        for index, scene in enumerate(normalized, start=1):
            scene["original_scene"] = scene.get("scene")
            scene["scene"] = index

    return normalized


def _merge_group(group: list[dict], fps: int) -> dict:
    if len(group) == 1:
        return deepcopy(group[0])

    start = float(group[0]["abs_start_seconds"])
    end = float(group[-1]["abs_end_seconds"])
    duration = end - start
    frame_count = frame_count_from_duration(duration, fps)

    anchor = max(group, key=lambda scene: int(scene["frame_count"]))
    merged = deepcopy(anchor)

    merged["abs_start_seconds"] = start
    merged["abs_end_seconds"] = end
    merged["duration_seconds"] = round(duration, 6)
    merged["frame_count"] = frame_count

    merged["metadata"] = deepcopy(anchor.get("metadata", {}))
    merged["metadata"]["merged_sources"] = [
        {
            "scene": scene["scene"],
            "abs_start_seconds": scene["abs_start_seconds"],
            "abs_end_seconds": scene["abs_end_seconds"],
            "duration_seconds": scene["duration_seconds"],
            "frame_count": scene["frame_count"],
            "type": scene.get("metadata", {}).get("type", ""),
        }
        for scene in group
    ]

    merged["ltx"] = deepcopy(anchor["ltx"])
    merged["z_image"] = deepcopy(anchor["z_image"])

    merged_relays = []
    cursor_frames = 0

    for scene in group:
        relays = scene.get("ltx", {}).get("prompt_relay", [])
        scene_frame_count = int(scene["frame_count"])
        scene_timeline_frames = max(1, scene_frame_count - 1)

        for relay in relays:
            new_relay = deepcopy(relay)
            new_relay["frame_start"] = cursor_frames + int(relay["frame_start"])
            new_relay["frame_end"] = cursor_frames + int(relay["frame_end"])
            new_relay["frame_start"] = max(0, min(new_relay["frame_start"], frame_count - 1))
            new_relay["frame_end"] = max(
                new_relay["frame_start"] + 1,
                min(new_relay["frame_end"], frame_count - 1),
            )
            merged_relays.append(new_relay)

        cursor_frames += scene_timeline_frames

    merged["ltx"]["prompt_relay"] = merged_relays
    return merged


def _split_render_scene_to_max(scene: dict, fps: int, max_duration: float) -> list[dict]:
    duration = float(scene["duration_seconds"])
    if duration <= max_duration:
        return [scene]

    parts = max(1, int(duration // max_duration))
    if duration / parts > max_duration:
        parts += 1

    part_duration = duration / parts
    result = []

    for i in range(parts):
        part = deepcopy(scene)

        start = float(scene["abs_start_seconds"]) + i * part_duration
        end = float(scene["abs_end_seconds"]) if i == parts - 1 else float(scene["abs_start_seconds"]) + (i + 1) * part_duration

        part["abs_start_seconds"] = start
        part["abs_end_seconds"] = end
        part["duration_seconds"] = round(end - start, 6)
        part["frame_count"] = frame_count_from_duration(end - start, fps)

        # Conservative relay fallback for split scenes.
        part["ltx"]["prompt_relay"] = [
            {
                "frame_start": 0,
                "frame_end": max(1, int(part["frame_count"]) - 1),
                "state": scene.get("metadata", {}).get("type", "instrumental"),
                "prompt": "continue the same motion and emotional direction from the scene",
            },
        ]

        result.append(part)

    return result


def normalize_render_plan_file(
    input_render_plan: str | Path,
    output_render_plan: str | Path,
    min_duration: float,
    max_duration: float,
    renumber: bool = True,
) -> Path:
    input_render_plan = coerce_local_path(input_render_plan)
    output_render_plan = coerce_local_path(output_render_plan)

    plan = json.loads(input_render_plan.read_text(encoding="utf-8"))
    normalized = normalize_render_plan(
        render_plan=plan,
        min_duration=min_duration,
        max_duration=max_duration,
        renumber=renumber,
    )

    output_render_plan.parent.mkdir(parents=True, exist_ok=True)
    output_render_plan.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return output_render_plan
