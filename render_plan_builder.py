from __future__ import annotations

from pathlib import Path
import json

from video_settings import VideoSettings


def _clamp_relay_segment(frame_start: int, frame_end: int, frame_count: int) -> tuple[int, int] | None:
    frame_start = max(0, min(frame_start, frame_count - 1))
    frame_end = max(frame_start + 1, min(frame_end, frame_count))

    if frame_end <= frame_start:
        return None

    return frame_start, frame_end


def build_render_plan(
    scene_prompts_json: str | Path,
    ltx_prompt_relay_json: str | Path,
    output_json_file: str | Path,
    video_settings: VideoSettings,
) -> Path:
    """
    Combines:
    - scene_prompts_json: per-scene Z-Image and LTX prompts
    - ltx_prompt_relay_json: per-scene frame-relative singing/instrumental relay

    Produces:
    - render_plan.json

    Rule:
    - One scene == one cut == one LTX render pass.
    - frame_count == round(fps * duration_seconds) + 1.
    """

    scene_prompts = json.loads(Path(scene_prompts_json).read_text(encoding="utf-8"))
    relay_scenes = json.loads(Path(ltx_prompt_relay_json).read_text(encoding="utf-8"))

    relay_by_scene = {int(scene["scene"]): scene for scene in relay_scenes}

    render_plan = []

    for scene in scene_prompts:
        scene_number = int(scene["scene"])
        relay_scene = relay_by_scene.get(scene_number)

        if relay_scene is None:
            raise ValueError(f"No relay data found for scene {scene_number}")

        duration_seconds = float(scene["duration"])
        frame_count = video_settings.scene_frame_count(duration_seconds)

        zimage_prompt = scene["zimage_prompt"]
        ltx_base_prompt = scene["ltx_base_prompt"]

        prompt_relay = []

        for relay in relay_scene.get("prompt_relay", []):
            clamped = _clamp_relay_segment(
                int(relay["frame_start"]),
                int(relay["frame_end"]),
                frame_count,
            )
            if clamped is None:
                continue

            frame_start, frame_end = clamped
            state = relay["state"]

            if state == "singing":
                state_prompt = (
                    "During this frame range, the same subject sings with expressive lip sync. "
                    "Preserve the same shot, lighting, character identity, wardrobe, and environment."
                )
                lyrics = relay.get("lyrics", "") or relay.get("text", "")
                if lyrics:
                    state_prompt += f" The lyrics being performed are: {lyrics}"
            else:
                state_prompt = (
                    "During this frame range, the same subject is silent. "
                    "No singing, no lip movement. Preserve the same shot, lighting, character identity, wardrobe, and environment."
                )

            prompt_relay.append({
                "frame_start": frame_start,
                "frame_end": frame_end,
                "state": state,
                "prompt": f"{ltx_base_prompt} {state_prompt}",
            })

        if not prompt_relay:
            state = "singing" if scene.get("type") == "vocals" else "instrumental"
            if state == "singing":
                state_prompt = (
                    "The same subject sings with expressive lip sync throughout the shot. "
                    "Preserve the same shot, lighting, character identity, wardrobe, and environment."
                )
            else:
                state_prompt = (
                    "The same subject is silent throughout the shot. "
                    "No singing, no lip movement. Preserve the same shot, lighting, character identity, wardrobe, and environment."
                )

            prompt_relay.append({
                "frame_start": 0,
                "frame_end": frame_count - 1,
                "state": state,
                "prompt": f"{ltx_base_prompt} {state_prompt}",
            })

        render_plan.append({
            "scene": scene_number,
            "cut": True,
            "abs_start_seconds": scene["start"],
            "abs_end_seconds": scene["end"],
            "duration_seconds": round(duration_seconds, 3),
            "fps": video_settings.fps,
            "width": video_settings.width,
            "height": video_settings.height,
            "frame_count": frame_count,
            "z_image": {"prompt": zimage_prompt, "frame": 0},
            "ltx": {"base_prompt": ltx_base_prompt, "prompt_relay": prompt_relay},
            "metadata": {
                "segment_id": scene["segment_id"],
                "type": scene["type"],
                "lyrics": scene.get("lyrics", ""),
                "base_concept": scene.get("base_concept", ""),
                "camera_motion": scene.get("camera_motion", ""),
                "character_motion": scene.get("character_motion", ""),
            },
        })

    output_json_file = Path(output_json_file)
    output_json_file.parent.mkdir(parents=True, exist_ok=True)
    output_json_file.write_text(json.dumps(render_plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_json_file
