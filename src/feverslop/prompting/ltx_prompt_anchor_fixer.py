from __future__ import annotations

import json
import re
from pathlib import Path

FORBIDDEN_RELAY_PHRASES = [
    "no subject visible",
    "no visible subject",
    "without the subject",
    "without the character",
    "tree sings",
    "bark sings",
    "shadows sing",
    "ropes sing",
    "branches sing",
    "faces in the bark sing",
    "weathered faces in the bark",
]


def _clean_text(value: str) -> str:
    return " ".join(str(value).replace("\n", " ").split()).strip()


def _sentence_limit(value: str, max_chars: int = 850) -> str:
    value = _clean_text(value)
    if len(value) <= max_chars:
        return value
    truncated = value[:max_chars].rsplit(" ", 1)[0].strip()
    sentence_end = max(truncated.rfind(". "), truncated.rfind("! "), truncated.rfind("? "))
    if sentence_end >= max_chars // 2:
        return truncated[:sentence_end + 1]
    while truncated and truncated[-1] in ".,;:!?":
        truncated = truncated.rstrip(".,;:!? ")
    return truncated


class LTXPromptAnchorFixer:
    """Makes render_plan LTX prompts safer for I2V.

    Problem this fixes:
    - Z-Image startframe shows the main subject.
    - LTX base_prompt or relay prompt cuts away to tree/bark/ropes/shadows.
    - LTX immediately replaces the startframe composition after 1-2 frames.

    Strategy:
    - For vocals/mixed scenes, anchor LTX base_prompt to the main subject and startframe composition.
    - For all scenes, preserve startframe composition.
    - For singing relay segments, force visible main subject + lip sync.
    - For instrumental relay segments, keep subject visible when the scene has vocals/mixed, but silent.
    - Remove bad phrases like "no subject visible to sing".
    """

    def __init__(
        self,
        subject_anchor: str,
        max_base_prompt_chars: int = 1200,
        max_relay_chars: int = 260,
    ):
        self.subject_anchor = _clean_text(subject_anchor)
        self.max_base_prompt_chars = max_base_prompt_chars
        self.max_relay_chars = max_relay_chars

    def fix_file(
        self,
        input_render_plan: str | Path,
        output_render_plan: str | Path,
    ) -> Path:
        input_render_plan = Path(input_render_plan)
        output_render_plan = Path(output_render_plan)

        plan = json.loads(input_render_plan.read_text(encoding="utf-8"))
        fixed = self.fix_render_plan(plan)

        output_render_plan.parent.mkdir(parents=True, exist_ok=True)
        output_render_plan.write_text(
            json.dumps(fixed, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_render_plan

    def fix_render_plan(self, render_plan: list[dict]) -> list[dict]:
        return [self.fix_scene(scene) for scene in render_plan]

    def fix_scene(self, scene: dict) -> dict:
        scene = dict(scene)
        scene["ltx"] = dict(scene["ltx"])

        scene_type = scene.get("metadata", {}).get("type", "")
        has_vocals = scene_type in {"vocals", "mixed"}

        scene["ltx"]["t2i_prompt"] = self._fix_t2i_prompt(scene)
        scene["ltx"]["base_prompt"] = self._fix_base_prompt(scene, has_vocals)
        fixed_i2v = self._fix_original_style_prompt(scene, has_vocals)
        scene["ltx"]["original_style_i2v_prompt"] = fixed_i2v
        scene["ltx"]["i2v_prompt_from_t2i"] = fixed_i2v

        relays = scene["ltx"].get("prompt_relay", [])
        scene["ltx"]["prompt_relay"] = [
            self._fix_relay(scene, relay, has_vocals)
            for relay in relays
        ]

        return scene

    def _scene_subject_anchor(self, scene: dict) -> str:
        actor_ids = [
            _clean_text(actor_id)
            for actor_id in scene.get("references", {}).get("actor_ids", [])
            if _clean_text(actor_id)
        ]
        if not actor_ids:
            return self.subject_anchor
        if len(actor_ids) == 1:
            return actor_ids[0]
        return ", ".join(actor_ids[:-1]) + f" and {actor_ids[-1]}"

    def _fix_t2i_prompt(self, scene: dict) -> str:
        prompt = _clean_text(scene["ltx"].get("t2i_prompt", ""))
        if not prompt:
            prompt = _clean_text(scene.get("z_image", {}).get("prompt", ""))
        if not prompt:
            prompt = _clean_text(scene["ltx"].get("base_prompt", ""))
        return _sentence_limit(prompt, self.max_base_prompt_chars)

    def _fix_base_prompt(self, scene: dict, has_vocals: bool) -> str:
        base = _clean_text(scene["ltx"].get("base_prompt", ""))
        z_prompt = _clean_text(scene["ltx"].get("t2i_prompt", "")) or _clean_text(scene.get("z_image", {}).get("prompt", ""))
        camera = _clean_text(scene.get("metadata", {}).get("camera_motion", ""))
        motion = _clean_text(scene.get("metadata", {}).get("character_motion", ""))
        concept = _clean_text(scene.get("metadata", {}).get("base_concept", ""))
        subject_anchor = self._scene_subject_anchor(scene)

        if has_vocals:
            prefix = (
                f"Start frame: {z_prompt}. "
                f"Lock the first frame to this exact composition and continue directly from it without fades, dissolves, crossfades, or shot changes. "
                f"Medium cinematic shot of {subject_anchor}, clearly visible in the foreground, "
                f"same identity, same wardrobe, same lighting, same location. "
                f"The main subject remains visible for lip sync throughout the shot. "
                f"Do not cut away to only the tree, bark, ropes, canopy, shadows, or macro details. "
            )
        else:
            prefix = (
                f"Start frame: {z_prompt}. "
                f"Lock the first frame to this exact composition and continue directly from it without fades, dissolves, crossfades, or shot changes. "
                f"Keep the same camera framing, same location, same lighting, and same visual continuity. "
            )

        dynamic = []
        if camera:
            dynamic.append(f"Camera motion: {camera}.")
        if motion:
            dynamic.append(f"Subject or environment motion: {motion}.")
        if concept:
            dynamic.append(f"Story beat: {concept}.")

        if has_vocals:
            dynamic.append(
                "Environmental motion happens around or behind the visible main subject, never replacing the subject.",
            )

        # Use base as secondary context, not as primary instruction.
        prompt = prefix + " ".join(dynamic) + " " + base

        return _sentence_limit(prompt, self.max_base_prompt_chars)

    def _fix_original_style_prompt(self, scene: dict, has_vocals: bool) -> str:
        z_prompt = _clean_text(scene["ltx"].get("t2i_prompt", "")) or _clean_text(scene.get("z_image", {}).get("prompt", ""))
        if not z_prompt:
            z_prompt = _clean_text(scene["ltx"].get("base_prompt", ""))
        # Reserve prompt space for performance and continuity instructions. Those
        # are operational requirements, while the startframe text is context.
        z_prompt = _sentence_limit(z_prompt, max(120, self.max_base_prompt_chars // 2))

        camera = _clean_text(scene.get("metadata", {}).get("camera_motion", ""))
        motion = _clean_text(scene.get("metadata", {}).get("character_motion", ""))
        concept = _clean_text(scene.get("metadata", {}).get("base_concept", ""))
        subject_anchor = self._scene_subject_anchor(scene)

        if has_vocals:
            verb = "remains" if len(scene.get("references", {}).get("actor_ids", [])) <= 1 else "remain"
            sing_verb = "sings" if verb == "remains" else "sing"
            performance = (
                f"{subject_anchor} {verb} clearly visible and {sing_verb} with controlled lip sync; "
                "the body stays in the same starting pose and framing."
            )
        else:
            performance = (
                f"{subject_anchor} remains clearly visible with a relaxed still mouth; "
                "only subtle breathing, fabric, hair, and atmospheric motion occur."
            )

        details = []
        if camera:
            details.append(f"Camera motion: {camera}.")
        if motion:
            details.append(f"Subject or environment motion: {motion}.")
        if concept:
            details.append(f"Story beat: {concept}.")

        prompt = (
            f"Start frame: {z_prompt}. "
            f"Lock the first frame to this exact composition and continue directly from it without fades, dissolves, crossfades, or shot changes. "
            f"{performance} {' '.join(details)} "
            "Keep the subject visible and centered, with the same framing, location, wardrobe, lighting, and subject identity. "
            "Do not cut away to a different shot or introduce a new action."
        )
        return _sentence_limit(prompt, self.max_base_prompt_chars)

    def _fix_relay(self, scene: dict, relay: dict, has_vocals: bool) -> dict:
        relay = dict(relay)
        state = relay.get("state", "")
        prompt = _clean_text(relay.get("prompt", ""))

        prompt = self._remove_forbidden(prompt)

        camera = _clean_text(scene.get("metadata", {}).get("camera_motion", ""))
        motion = _clean_text(scene.get("metadata", {}).get("character_motion", ""))
        subject_anchor = self._scene_subject_anchor(scene)
        subject_verb = "remains" if len(scene.get("references", {}).get("actor_ids", [])) <= 1 else "remain"

        if state == "singing":
            fixed = (
                f"{subject_anchor} {subject_verb} clearly visible in the foreground, "
                f"singing with controlled lip sync and focused emotion; "
                "do not fade, dissolve, crossfade, or cut away to a different shot"
            )
            if camera:
                fixed += f", {camera}"
            if motion:
                fixed += f", {motion}"
            fixed += "; tree, ropes, shadows, fog, and branches move only behind or around the subject"
        elif has_vocals:
            fixed = (
                f"{subject_anchor} {subject_verb} clearly visible in the foreground, silent with no lip movement; "
                "do not fade, dissolve, crossfade, or cut away to a different shot"
            )
            if camera:
                fixed += f", {camera}"
            if motion:
                fixed += f", {motion}"
            fixed += "; environmental motion stays behind the subject and does not take over the shot"
        else:
            # Instrumental-only scenes may focus on environment, but still avoid scene jumps.
            if prompt:
                fixed = prompt
            else:
                fixed = "preserve the same shot, subtle atmospheric motion, no sudden scene change"
            fixed += "; preserve the startframe composition and visual continuity, with no fade or shot change"

        relay["prompt"] = _sentence_limit(fixed, self.max_relay_chars)
        return relay

    @staticmethod
    def _remove_forbidden(prompt: str) -> str:
        result = prompt
        for phrase in FORBIDDEN_RELAY_PHRASES:
            result = re.sub(re.escape(phrase), "", result, flags=re.IGNORECASE)
        result = re.sub(r"\s+", " ", result).strip(" ,.;")
        return result


def validate_anchor_file(
    render_plan_path: str | Path,
    subject_hint: str,
) -> list[str]:
    plan = json.loads(Path(render_plan_path).read_text(encoding="utf-8"))
    subject_words = [w.lower() for w in re.findall(r"[A-Za-z]+", subject_hint) if len(w) >= 4]
    warnings = []

    for scene in plan:
        scene_no = scene.get("scene")
        scene_type = scene.get("metadata", {}).get("type", "")
        base = scene.get("ltx", {}).get("base_prompt", "").lower()

        if scene_type in {"vocals", "mixed"}:
            if not any(word in base for word in subject_words[:8]):
                warnings.append(f"Scene {scene_no}: vocal/mixed base_prompt may not mention subject anchor.")

        for relay in scene.get("ltx", {}).get("prompt_relay", []):
            prompt = relay.get("prompt", "").lower()
            state = relay.get("state", "")
            if state == "singing":
                if "sing" not in prompt and "lip sync" not in prompt and "lip-sync" not in prompt:
                    warnings.append(f"Scene {scene_no}: singing relay lacks singing/lip-sync wording.")
                if "no subject visible" in prompt:
                    warnings.append(f"Scene {scene_no}: singing relay contains forbidden no-subject phrase.")

    return warnings
