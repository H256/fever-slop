from __future__ import annotations

from pathlib import Path
import json

from video_settings import VideoSettings


def build_render_plan(
    final_scene_prompts_json: str | Path,
    ltx_prompt_relay_json: str | Path,
    output_json_file: str | Path,
    video_settings: VideoSettings,
) -> Path:
    final_scenes = json.loads(Path(final_scene_prompts_json).read_text(encoding="utf-8"))
    relay_scenes = json.loads(Path(ltx_prompt_relay_json).read_text(encoding="utf-8"))

    relay_by_scene = {
        int(scene["scene"]): scene
        for scene in relay_scenes
    }

    render_plan = []

    for scene in final_scenes:
        scene_number = int(scene["scene"])
        relay_scene = relay_by_scene.get(scene_number)

        if relay_scene is None:
            raise ValueError(f"No relay data found for scene {scene_number}")

        duration_seconds = float(scene["duration"])

        # IMPORTANT RULE:
        # total_frames = fps * duration_seconds + 1
        frame_count = video_settings.seconds_to_frame(duration_seconds) + 1

        base_prompt = scene["final_prompt"]

        z_image_prompt = (
            f"{base_prompt}. "
            "single cinematic keyframe, sharp detailed still image, "
            "no motion blur, strong composition, suitable as the first frame of a video shot"
        )

        prompt_relay = []

        for relay in relay_scene["prompt_relay"]:
            frame_start = int(relay["frame_start"])
            frame_end = int(relay["frame_end"])

            # Clamp into scene frame range
            frame_start = max(0, min(frame_start, frame_count - 1))
            frame_end = max(frame_start + 1, min(frame_end, frame_count))

            state = relay["state"]
            relay_prompt = relay["prompt"]

            if state == "singing":
                final_relay_prompt = (
                    f"{base_prompt}. "
                    f"{relay_prompt}. "
                    "the character sings with expressive lip sync"
                )
            else:
                final_relay_prompt = (
                    f"{base_prompt}. "
                    f"{relay_prompt}. "
                    "the character does not sing, no lip movement"
                )

            prompt_relay.append({
                "frame_start": frame_start,
                "frame_end": frame_end,
                "state": state,
                "prompt": final_relay_prompt,
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

            "z_image": {
                "prompt": z_image_prompt,
                "frame": 0,
            },

            "ltx": {
                "base_prompt": base_prompt,
                "prompt_relay": prompt_relay,
            },

            "metadata": {
                "segment_id": scene["segment_id"],
                "type": scene["type"],
                "lyrics": scene.get("lyrics", ""),
                "base_concept": scene.get("base_concept", ""),
                "camera_motion": scene.get("camera_motion", ""),
                "character_motion": scene.get("character_motion", ""),
            }
        })

    output_json_file = Path(output_json_file)
    output_json_file.parent.mkdir(parents=True, exist_ok=True)

    output_json_file.write_text(
        json.dumps(render_plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return output_json_file