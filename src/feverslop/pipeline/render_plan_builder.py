from __future__ import annotations

from pathlib import Path
import random

from feverslop.ports.artifacts import ArtifactStore
from feverslop.config.video_settings import VideoSettings
from feverslop.path_utils import coerce_local_path
from feverslop.errors import FeverSlopDataError


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


def _scene_silent_mode(scene: dict) -> bool:
    return bool(scene.get("silent_mode") or (scene.get("metadata") or {}).get("silent_mode"))


def _contains_vocal_performance_prompt(value: str) -> bool:
    lower = str(value or "").lower()
    return any(token in lower for token in ("sings", "singing", "lip sync", "lip-sync", "lip-syncing", "belts out"))


def _effective_relay_state(state: object, scene: dict) -> str:
    if _scene_silent_mode(scene):
        return "instrumental"
    return str(state or "").strip().lower()


def _scene_references(scene: dict) -> dict:
    references = dict(scene.get("references") or {})
    actor_ids = list(references.get("actor_ids") or [])
    if len(actor_ids) > 4:
        raise ValueError(f"Scene {scene.get('scene')} references at most 4 actors for ltx_msr")
    if actor_ids:
        references["actor_ids"] = [str(actor_id) for actor_id in actor_ids]
    if "location_id" in references and references["location_id"] is not None:
        references["location_id"] = str(references["location_id"])
    return references


def _project_relative_path(path: str | Path, project_dir: Path) -> str:
    return (
        coerce_local_path(path, base_dir=project_dir, containment_root=project_dir)
        .resolve()
        .relative_to(project_dir.resolve())
        .as_posix()
    )


def _portable_audio_references(references: dict, project_dir: Path) -> None:
    if "reference_audio_paths" in references:
        references["reference_audio_paths"] = [
            _project_relative_path(path, project_dir)
            for path in references["reference_audio_paths"]
        ]
    if "_stem_audio_tags" in references:
        references["_stem_audio_tags"] = {
            _project_relative_path(path, project_dir): description
            for path, description in references["_stem_audio_tags"].items()
        }


def _require_key(d: dict[str, object], key: str, context: str) -> object:
    """Return d[key], raising FeverSlopDataError with full context if missing."""
    if key not in d:
        raise FeverSlopDataError(f"Missing key {key!r} in {context}")
    return d[key]


def build_original_style_i2v_prompt(scene: dict, seed: int = 0) -> str:
    scene_number = int(scene["scene"])
    scene_type = str(scene.get("type", "")).strip().lower()
    silent_mode = _scene_silent_mode(scene)
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
    if explicit_i2v_prompt and not (_scene_silent_mode(scene) and _contains_vocal_performance_prompt(explicit_i2v_prompt)):
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

    if scene_type == "vocals" and not silent_mode:
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
            "silent with no vocal performance, no lip movement, and the mouth relaxed and still."
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
    *,
    artifact_store: ArtifactStore,
    h3_prompts_json: str | Path | None = None,
    stem_list: list[str] | None = None,
    input_audio: Path | None = None,
    stem_files: dict[str, Path] | None = None,
    project_dir: Path | None = None,
    seed: int = 0,
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

    scene_prompts = artifact_store.read_json(scene_prompts_json)
    relay_scenes = artifact_store.read_json(ltx_prompt_relay_json)

    relay_by_scene = {int(_require_key(scene, "scene", "relay scene data")): scene for scene in relay_scenes}

    h3_by_segment: dict[str, dict] = {}
    if h3_prompts_json is not None:
        h3_prompts = artifact_store.read_json(h3_prompts_json)
        h3_by_segment = {str(item.get("segment_id", "")): item for item in h3_prompts}

    render_plan = []

    for scene in scene_prompts:
        scene_number = int(_require_key(scene, "scene", "scene prompts"))
        scene_seed = (
            random.SystemRandom().randint(0, 2**63 - 1)
            if int(seed) == -1
            else int(seed)
        )
        relay_scene = relay_by_scene.get(scene_number)

        if relay_scene is None:
            raise ValueError(f"No relay data found for scene {scene_number}")

        duration_seconds = float(_require_key(scene, "duration", f"scene prompts, scene {scene_number}"))
        frame_count = video_settings.scene_frame_count_between(
            float(_require_key(scene, "start", f"scene prompts, scene {scene_number}")),
            float(_require_key(scene, "end", f"scene prompts, scene {scene_number}")),
        )

        zimage_prompt = _require_key(scene, "zimage_prompt", f"scene prompts, scene {scene_number}")
        t2i_prompt = str(scene.get("t2i_prompt") or scene.get("zimage_prompt") or scene.get("ltx_base_prompt") or scene.get("base_prompt") or "").strip()
        ltx_base_prompt = t2i_prompt
        original_style_i2v_prompt = build_original_style_i2v_prompt(scene, seed=scene_seed)
        i2v_prompt_from_t2i = original_style_i2v_prompt if _scene_silent_mode(scene) else str(
            scene.get("i2v_prompt_from_t2i")
            or scene.get("original_style_i2v_prompt")
            or original_style_i2v_prompt
        ).strip()

        prompt_relay = []

        for relay in relay_scene.get("prompt_relay", []):
            clamped = _clamp_relay_segment(
                int(_require_key(relay, "frame_start", f"relay data, scene {scene_number}")),
                int(_require_key(relay, "frame_end", f"relay data, scene {scene_number}")),
                frame_count,
            )
            if clamped is None:
                continue

            frame_start, frame_end = clamped
            state = _effective_relay_state(_require_key(relay, "state", f"relay data, scene {scene_number}"), scene)

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
                    "No vocal performance, mouth closed, no lip movement. Preserve the same shot, lighting, character identity, wardrobe, and environment."
                )

            prompt_relay.append({
                "frame_start": frame_start,
                "frame_end": frame_end,
                "state": state,
                "prompt": f"{ltx_base_prompt} {state_prompt}",
            })

        if not prompt_relay:
            state = "singing" if scene.get("type") == "vocals" and not _scene_silent_mode(scene) else "instrumental"
            if state == "singing":
                state_prompt = (
                    "The same subject sings with expressive lip sync throughout the shot. "
                    "Preserve the same shot, lighting, character identity, wardrobe, and environment."
                )
            else:
                state_prompt = (
                    "The same subject is silent throughout the shot. "
                    "No vocal performance, mouth closed, no lip movement. Preserve the same shot, lighting, character identity, wardrobe, and environment."
                )

            prompt_relay.append({
                "frame_start": 0,
                "frame_end": frame_count - 1,
                "state": state,
                "prompt": f"{ltx_base_prompt} {state_prompt}",
            })

        render_scene = {
            "scene": scene_number,
            "seed": scene_seed,
            "cut": True,
            "abs_start_seconds": _require_key(scene, "start", f"scene prompts, scene {scene_number}"),
            "abs_end_seconds": _require_key(scene, "end", f"scene prompts, scene {scene_number}"),
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
                "segment_id": _require_key(scene, "segment_id", f"scene prompts, scene {scene_number}"),
                "type": _require_key(scene, "type", f"scene prompts, scene {scene_number}"),
                "silent_mode": _scene_silent_mode(scene),
                "lyrics": scene.get("lyrics", ""),
                "base_concept": scene.get("base_concept", ""),
                "camera_motion": scene.get("camera_motion", ""),
                "character_motion": scene.get("character_motion", ""),
            },
        }
        references = _scene_references(scene)
        if references and project_dir is not None:
            _portable_audio_references(references, project_dir)
        if references:
            render_scene["references"] = references
        h3_entry = h3_by_segment.get(scene.get("segment_id", ""))
        if h3_entry and h3_entry.get("prompt"):
            render_scene["h3"] = {"prompt": str(h3_entry["prompt"]).strip()}
        # -- stem audio paths (MiniMax H3 R2V) --
        if stem_list is not None and stem_files is not None:
            # Priority ordering: lip-sync-critical stems first, so they occupy
            # the first audio reference slots in the workflow.
            priority_stems = ["vocals", "full_mix"]
            priority_first = [s for s in priority_stems if s in stem_list]
            for s in stem_list:
                if s not in priority_first:
                    priority_first.append(s)
            resolved_stem_paths: dict[str, str] = {}
            for stem_name in priority_first:
                if stem_name == "full_mix" and input_audio is not None:
                    stem_path = input_audio
                elif stem_name in stem_files:
                    stem_path = stem_files[stem_name]
                else:
                    continue
                if project_dir is not None:
                    resolved_stem_paths[stem_name] = _project_relative_path(stem_path, project_dir)
                else:
                    resolved_stem_paths[stem_name] = stem_path.as_posix()
                # missing stems silently skipped
            if resolved_stem_paths:
                render_scene["stem_audio"] = {
                    "stems": list(resolved_stem_paths.keys()),
                    "paths": resolved_stem_paths,
                }
                # Merge stem audio into reference_audio_paths so prompt generators
                # (H3, R2V style builder) can produce <Audio N> tags for stems.
                refs = render_scene.setdefault("references", {})
                existing: list[str] = list(refs.get("reference_audio_paths", []))
                stem_tag_map = {
                    "vocals": "audio_transfer - vocal singing lip-synced to the audio signal",
                    "full_mix": "full_mix - original song for beat and rhythm continuity",
                    "drums": "drums stem",
                    "bass": "bass stem",
                    "other": "other stem",
                }
                seen: set[str] = set(existing)
                merged_audio: list[str] = list(existing)
                for stem_name, stem_path in resolved_stem_paths.items():
                    if stem_path not in seen:
                        merged_audio.append(stem_path)
                        seen.add(stem_path)
                refs["reference_audio_paths"] = merged_audio
                refs["_stem_audio_tags"] = {
                    stem_path: stem_tag_map.get(stem_name, f"{stem_name} stem")
                    for stem_name, stem_path in resolved_stem_paths.items()
                }
        render_plan.append(render_scene)

    return artifact_store.write_json(output_json_file, render_plan)
