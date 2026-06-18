from __future__ import annotations

from pathlib import Path
import json
import random

from video_settings import VideoSettings


CAMERA_MOTION_DETAILS = [
    "slow forward dolly",
    "gentle handheld drift",
    "subtle arc around the subject",
    "locked medium close-up with slight push-in",
    "soft vertical crane rise",
    "steady shoulder-level tracking",
]

VOCAL_CHARACTER_MOTION_DETAILS = [
    "expressive singing posture",
    "natural breath-led head movement",
    "hands move with the rhythm",
    "shoulders sway subtly while performing",
    "eyes stay engaged with the camera",
    "controlled microphone performance gestures",
]

INSTRUMENTAL_CHARACTER_MOTION_DETAILS = [
    "calm breathing and small posture shifts",
    "hands relax naturally at the sides",
    "eyes scan the environment quietly",
    "subtle weight shift in place",
    "still mouth with restrained facial movement",
    "slow turn of the shoulders",
]

LIGHTING_DETAILS = [
    "soft rim light outlining the subject",
    "practical lights glowing in the background",
    "high contrast stage lighting",
    "diffused window light across the face",
    "colored reflections moving gently through the scene",
    "low-key cinematic light with clear facial detail",
]

TIME_OF_DAY_DETAILS = [
    "late night",
    "blue hour",
    "golden hour",
    "early morning",
    "deep evening",
    "overcast afternoon",
]

WEATHER_DETAILS = [
    "clear air",
    "light haze",
    "soft rain outside the frame",
    "misty atmosphere",
    "dry still air",
    "faint drifting dust",
]

FACIAL_EXPRESSION_DETAILS = [
    "focused eyes",
    "subtle intensity",
    "restrained vulnerability",
    "serene concentration",
    "quiet confidence",
    "emotional resolve",
]

EMOTION_DETAILS = [
    "intimate",
    "urgent",
    "melancholic",
    "defiant",
    "tender",
    "haunted",
]

LOCATION_DETAILS = [
    "within the established setting",
    "in the same framed environment",
    "without changing location",
    "inside the existing scene geography",
]


class DetailListPicker:
    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self._cycles: dict[str, list[str]] = {}
        self._positions: dict[str, int] = {}

    def pick(
        self,
        list_name: str,
        items: list[str],
        scene_number: int,
        strategy: str = "random",
        index: int | None = None,
        pick_count: int = 1,
    ) -> str:
        if not items:
            return ""

        count = max(1, int(pick_count))

        if strategy == "index":
            start = int(index if index is not None else scene_number - 1)
            picks = [items[(start + offset) % len(items)] for offset in range(count)]
        elif strategy == "random_no_repeat":
            picks = [self._pick_no_repeat(list_name, items) for _ in range(count)]
        elif strategy == "random":
            rng = random.Random(f"{self.seed}:{scene_number}:{list_name}")
            picks = [items[rng.randrange(len(items))] for _ in range(count)]
        else:
            raise ValueError("strategy must be 'index', 'random', or 'random_no_repeat'")

        if len(picks) == 1:
            return picks[0]
        return f"start with {picks[0]} then follow with {picks[1]}"

    def _pick_no_repeat(self, list_name: str, items: list[str]) -> str:
        position = self._positions.get(list_name, 0)
        cycle = self._cycles.get(list_name)

        if cycle is None or position >= len(cycle):
            cycle = list(items)
            random.Random(f"{self.seed}:{list_name}:{position // max(1, len(items))}").shuffle(cycle)
            self._cycles[list_name] = cycle
            position = 0

        value = cycle[position]
        self._positions[list_name] = position + 1
        return value


def _clamp_relay_segment(frame_start: int, frame_end: int, frame_count: int) -> tuple[int, int] | None:
    frame_start = max(0, min(frame_start, frame_count - 1))
    frame_end = max(frame_start + 1, min(frame_end, frame_count))

    if frame_end <= frame_start:
        return None

    return frame_start, frame_end


def _relay_states(prompt_relay: list[dict]) -> set[str]:
    return {str(relay.get("state", "")).strip().lower() for relay in prompt_relay if relay.get("state")}


def _render_mode_hint(scene_type: str, prompt_relay: list[dict]) -> str:
    scene_type = scene_type.strip().lower()
    states = _relay_states(prompt_relay)
    if len(states) > 1:
        return "relay"
    if scene_type == "mixed":
        return "relay"
    return "single_prompt"


def build_original_style_i2v_prompt(scene: dict, seed: int = 0) -> str:
    scene_number = int(scene["scene"])
    scene_type = str(scene.get("type", "")).strip().lower()
    zimage_prompt = str(
        scene.get("t2i_prompt")
        or scene.get("zimage_prompt")
        or scene.get("z_image", {}).get("prompt", "")
        or ""
    ).strip()
    explicit_i2v_prompt = str(
        scene.get("i2v_prompt_from_t2i")
        or scene.get("original_style_i2v_prompt")
        or ""
    ).strip()
    if explicit_i2v_prompt:
        return explicit_i2v_prompt

    base_prompt = str(
        scene.get("t2i_prompt")
        or scene.get("zimage_prompt")
        or scene.get("ltx_base_prompt")
        or scene.get("base_prompt")
        or ""
    ).strip()
    base_concept = str(scene.get("base_concept", "")).strip()
    visual_foundation = zimage_prompt or base_prompt

    picker = DetailListPicker(seed=seed)
    camera_motion = picker.pick("camera_motion", CAMERA_MOTION_DETAILS, scene_number, "random")
    lighting = picker.pick("lighting", LIGHTING_DETAILS, scene_number, "random")
    time_of_day = picker.pick("time_of_day", TIME_OF_DAY_DETAILS, scene_number, "random")
    weather = picker.pick("weather", WEATHER_DETAILS, scene_number, "random")
    facial_expression = picker.pick("facial_expression", FACIAL_EXPRESSION_DETAILS, scene_number, "random")
    emotion = picker.pick("emotion", EMOTION_DETAILS, scene_number, "random")
    location = picker.pick("location", LOCATION_DETAILS, scene_number, "index")

    if scene_type == "vocals":
        character_motion = picker.pick(
            "vocal_character_motion",
            VOCAL_CHARACTER_MOTION_DETAILS,
            scene_number,
            "random",
            pick_count=2,
        )
        performance = (
            "The visible subject sings with expressive lip sync throughout the shot, "
            "with the mouth performance matching the vocal energy."
        )
    else:
        character_motion = picker.pick(
            "instrumental_character_motion",
            INSTRUMENTAL_CHARACTER_MOTION_DETAILS,
            scene_number,
            "random",
            pick_count=2,
        )
        performance = (
            "The visible subject remains present and framed throughout the shot, "
            "with the mouth relaxed and still."
        )

    identity = f"Scene identity: {base_concept}." if base_concept else "Scene identity stays unchanged."
    return (
        f"Start frame: {visual_foundation}. "
        f"Lock the first frame to this exact composition and continue directly from it without fades, dissolves, crossfades, or shot changes. "
        f"{performance} {identity} Keep the subject visible and clearly framed. "
        f"Camera motion: {camera_motion}. Character motion: {character_motion}. "
        f"Lighting: {lighting}. Time of day: {time_of_day}. Weather and atmosphere: {weather}. "
        f"Facial expression: {facial_expression}. Mood: {emotion}. Location: {location}. "
        "Preserve the same pose, setting, outfit, subject identity, "
        "composition, and atmosphere. Do not introduce new characters, new locations, or new story events."
    )


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
    - frame_count == round(fps * duration_seconds).
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
        frame_count = video_settings.scene_frame_count_between(
            float(scene["start"]),
            float(scene["end"]),
        )

        zimage_prompt = scene["zimage_prompt"]
        t2i_prompt = str(scene.get("t2i_prompt") or scene.get("zimage_prompt") or scene.get("ltx_base_prompt") or scene.get("base_prompt") or "").strip()
        ltx_base_prompt = t2i_prompt
        i2v_prompt_from_t2i = str(
            scene.get("i2v_prompt_from_t2i")
            or scene.get("original_style_i2v_prompt")
            or build_original_style_i2v_prompt(scene, seed=video_settings.fps)
        ).strip()
        original_style_i2v_prompt = build_original_style_i2v_prompt(scene, seed=video_settings.fps)

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
            "ltx": {
                "base_prompt": ltx_base_prompt,
                "t2i_prompt": t2i_prompt,
                "i2v_prompt_from_t2i": i2v_prompt_from_t2i,
                "prompt_relay": prompt_relay,
                "original_style_i2v_prompt": original_style_i2v_prompt,
                "render_mode_hint": _render_mode_hint(str(scene.get("type", "")), prompt_relay),
            },
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
