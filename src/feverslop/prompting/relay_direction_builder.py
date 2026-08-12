from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any

from feverslop.ports.llm import LLMPort


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    original = text
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    start = text.find("[")
    if start == -1:
        raise ValueError(f"No JSON array start found in LLM response:\n{original}")

    end = text.rfind("]")

    if end == -1:
        candidate = text[start:].strip().rstrip(",") + "]"
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
            f"Original response:\n{original}\n\nCandidate:\n{candidate}"
        )


def _clean_direction(text: str, max_chars: int = 220) -> str:
    text = " ".join(str(text).replace("\n", " ").split())
    text = text.strip(" -|")
    text = re.sub(r"\bno subject visible\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bno visible subject\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ,.;")
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()
    return text


class RelayDirectionBuilder:
    def __init__(
        self,
        llm: LLMPort,
        max_words: int = 28,
        subject_anchor: str = (
            "the old weary warrior man with weathered scarred face, "
            "salt-and-pepper beard, tattered leather armor, and heavy frayed cloak"
        ),
    ):
        self.llm = llm
        self.max_words = max_words
        self.subject_anchor = subject_anchor

    def compact_render_plan_file(self, input_render_plan: str | Path, output_render_plan: str | Path) -> Path:
        input_render_plan = Path(input_render_plan)
        output_render_plan = Path(output_render_plan)

        plan = json.loads(input_render_plan.read_text(encoding="utf-8"))
        compacted = self.compact_render_plan(plan)

        output_render_plan.parent.mkdir(parents=True, exist_ok=True)
        output_render_plan.write_text(json.dumps(compacted, ensure_ascii=False, indent=2), encoding="utf-8")
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
        scene_type = scene.get("metadata", {}).get("type", "")
        has_vocals = scene_type in {"vocals", "mixed"}

        payload = {
            "scene": scene.get("scene"),
            "segment_id": scene.get("metadata", {}).get("segment_id"),
            "scene_type": scene_type,
            "lyrics": scene.get("metadata", {}).get("lyrics", ""),
            "main_subject": self.subject_anchor,
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
You create compact PromptRelay local directions for LTX image-to-video.

The global prompt is already provided separately. Do NOT repeat the full scene, style, location, lighting, or full story.

Return ONLY valid JSON array with exactly one object per relay segment:
[
  {{"index": 0, "prompt": "short local direction"}}
]

Hard rules:
- Keep each prompt under {self.max_words} words.
- Preserve every concrete action, manipulated object, and required prop from
  current_prompt (for example, cooking a rabbit); do not replace it with only
  atmosphere, composition, or camera language.
- current_prompt may be image-like, but treat its concrete action and object
  details as binding source material and translate them into observable motion.
- For state "singing": the main subject must remain clearly visible and must sing/lip-sync.
- Main subject: {self.subject_anchor}
- Never write: "no subject visible", "no visible subject", "tree sings", "bark sings", "shadows sing", "ropes sing".
- The tree, bark, shadows, fog, ropes, branches, spirits, or secondary figures must not sing.
- For instrumental: no singing and no lip movement.
- For vocals/mixed scenes: keep the main subject foreground or clearly visible; environment moves behind or around him.
- Make directions dynamic: turns, reaches, recoils, breathes, gazes, camera drifts, fog curls, light pulses.
- No frame numbers, no markdown, no comments.
""".strip()

        response = self.llm.complete_prompt(
            system_prompt=system_prompt,
            prompt=json.dumps(payload, ensure_ascii=False, indent=2),
        )

        try:
            items = _extract_json_array(response)
            by_index = {
                int(item["index"]): _clean_direction(item["prompt"])
                for item in items
                if "index" in item and "prompt" in item
            }
        except Exception:
            by_index = {}

        result = []

        for idx, relay in enumerate(relays):
            new_relay = dict(relay)
            source_prompt = _clean_direction(relay.get("prompt", ""), max_chars=500)
            if source_prompt:
                new_relay["source_prompt"] = source_prompt
            fallback = self._fallback_direction(scene, relay, has_vocals)
            prompt = by_index.get(idx, fallback)
            prompt = self._safety_fix_prompt(scene, relay, prompt, has_vocals)
            new_relay["prompt"] = prompt
            result.append(new_relay)

        return result

    def _safety_fix_prompt(self, scene: dict, relay: dict, prompt: str, has_vocals: bool) -> str:
        state = relay.get("state", "")
        p = _clean_direction(prompt)

        bad_patterns = [
            "no subject visible",
            "no visible subject",
            "tree sings",
            "bark sings",
            "shadows sing",
            "ropes sing",
            "branches sing",
        ]

        if any(pattern in p.lower() for pattern in bad_patterns):
            return self._fallback_direction(scene, relay, has_vocals)

        # If singing but no explicit visible subject/lip-sync intent, repair.
        if state == "singing":
            lower = p.lower()
            if "sing" not in lower and "lip sync" not in lower and "lip-sync" not in lower:
                return self._fallback_direction(scene, relay, has_vocals)
            if "warrior" not in lower and "man" not in lower and "subject" not in lower:
                return self._fallback_direction(scene, relay, has_vocals)

        if has_vocals and state != "singing":
            lower = p.lower()
            if "warrior" not in lower and "subject" not in lower and "man" not in lower:
                p = f"{self.subject_anchor} remains visible and silent, {p}"

        return _clean_direction(p)

    def _fallback_direction(self, scene: dict, relay: dict, has_vocals: bool) -> str:
        state = relay.get("state", "")
        camera = scene.get("metadata", {}).get("camera_motion", "")
        motion = scene.get("metadata", {}).get("character_motion", "")

        parts = []

        if state == "singing":
            parts.append(f"{self.subject_anchor} remains clearly visible in foreground, singing with controlled lip sync and focused emotion")
        elif has_vocals:
            parts.append(f"{self.subject_anchor} remains clearly visible in foreground, silent with no lip movement")
        else:
            parts.append("preserve the same shot and startframe composition")

        if camera:
            parts.append(camera)
        if motion:
            parts.append(motion)

        parts.append("environmental motion stays behind or around the subject")

        return _clean_direction(", ".join(parts))
