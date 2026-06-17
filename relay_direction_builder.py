from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any

from llm_client import LocalOpenAIClient


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in LLM response:\n{text}")

    return json.loads(text[start:end + 1])


def _clean_direction(text: str, max_chars: int = 220) -> str:
    text = " ".join(str(text).replace("\n", " ").split())
    text = text.strip(" -|")
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()
    return text


class RelayDirectionBuilder:
    """
    Converts verbose render_plan ltx.prompt_relay prompts into compact PromptRelay directions.

    Important:
    - global prompt remains scene["ltx"]["base_prompt"]
    - local relay prompts become short direction/motion instructions only
    - frame_start / frame_end / state are preserved
    """

    def __init__(
        self,
        llm: LocalOpenAIClient,
        max_words: int = 28,
    ):
        self.llm = llm
        self.max_words = max_words

    def compact_render_plan_file(
        self,
        input_render_plan: str | Path,
        output_render_plan: str | Path,
    ) -> Path:
        input_render_plan = Path(input_render_plan)
        output_render_plan = Path(output_render_plan)

        plan = json.loads(input_render_plan.read_text(encoding="utf-8"))
        compacted = self.compact_render_plan(plan)

        output_render_plan.parent.mkdir(parents=True, exist_ok=True)
        output_render_plan.write_text(
            json.dumps(compacted, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return output_render_plan

    def compact_render_plan(self, render_plan: list[dict]) -> list[dict]:
        result = []

        for scene in render_plan:
            new_scene = dict(scene)
            ltx = dict(scene["ltx"])
            relays = list(ltx.get("prompt_relay", []))

            if relays:
                compact_relays = self.compact_scene_relays(scene, relays)
                ltx["prompt_relay"] = compact_relays

            new_scene["ltx"] = ltx
            result.append(new_scene)

        return result

    def compact_scene_relays(self, scene: dict, relays: list[dict]) -> list[dict]:
        payload = {
            "scene": scene.get("scene"),
            "segment_id": scene.get("metadata", {}).get("segment_id"),
            "scene_type": scene.get("metadata", {}).get("type"),
            "lyrics": scene.get("metadata", {}).get("lyrics", ""),
            "base_prompt": scene["ltx"]["base_prompt"],
            "camera_motion": scene.get("metadata", {}).get("camera_motion", ""),
            "character_motion": scene.get("metadata", {}).get("character_motion", ""),
            "relay_segments": [
                {
                    "index": idx,
                    "frame_start": int(relay["frame_start"]),
                    "frame_end": int(relay["frame_end"]),
                    "state": relay.get("state", ""),
                    "current_prompt": relay.get("prompt", ""),
                }
                for idx, relay in enumerate(relays)
            ],
        }

        system_prompt = f"""
You create compact PromptRelay local directions for LTX video.

The global prompt is already provided separately. Do NOT repeat the global scene, subject, style, location, lighting, or full story.

Return ONLY valid JSON array with exactly one object per relay segment:
[
  {{"index": 0, "prompt": "short local direction"}}
]

Rules:
- Keep each prompt under {self.max_words} words.
- Each prompt should be only local direction: camera motion, character movement, interaction, expression, emotion, singing state, environmental motion.
- Do NOT repeat the full scene description.
- Do NOT include frame numbers.
- Do NOT include JSON comments or markdown.
- Do NOT mention "global prompt".
- For state "singing": the resolved main subject is the only one who sings or lip-syncs.
- For state "instrumental": no singing, no lip movement.
- If spirits, villagers, shadows, trees, or secondary figures appear, they must not sing unless explicitly the main subject.
- Make the directions dynamic and cinematic: turns, reaches, recoils, breathes, gazes, camera drifts, fog curls, light pulses.
- Prefer strong verbs.
- Avoid long descriptive adjectives.
""".strip()

        response = self.llm.complete_prompt(
            system_prompt=system_prompt,
            prompt=json.dumps(payload, ensure_ascii=False, indent=2),
        )

        items = _extract_json_array(response)
        by_index = {int(item["index"]): _clean_direction(item["prompt"]) for item in items}

        result = []

        for idx, relay in enumerate(relays):
            new_relay = dict(relay)
            fallback = self._fallback_direction(scene, relay)
            new_relay["prompt"] = by_index.get(idx, fallback)
            result.append(new_relay)

        return result

    @staticmethod
    def _fallback_direction(scene: dict, relay: dict) -> str:
        state = relay.get("state", "")
        camera = scene.get("metadata", {}).get("camera_motion", "")
        motion = scene.get("metadata", {}).get("character_motion", "")

        parts = []

        if camera:
            parts.append(camera)

        if motion:
            parts.append(motion)

        if state == "singing":
            parts.append("main subject sings with controlled lip sync and focused emotion")
        else:
            parts.append("main subject remains silent, no lip movement")

        return _clean_direction(", ".join(parts))
