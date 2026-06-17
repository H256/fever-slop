from __future__ import annotations

from pathlib import Path
import json
import re

from llm_client import LocalOpenAIClient
from music_video_prompt_style import (
    build_i2v_system_prompt,
    build_t2i_system_prompt,
    build_video_payload,
)


def clean_llm_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:text)?", "", text.strip(), flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text.strip()).strip()
    return text.strip()


class ScenePromptBuilder:
    """
    Builds model-specific prompts per scene.

    Important design:
    - Subject is injected into EVERY scene-generation call.
    - Z-Image prompts are still-image keyframe prompts.
    - LTX prompts are video prompts and may contain motion.
    """

    def __init__(self, llm: LocalOpenAIClient):
        self.llm = llm

    def build_zimage_prompt(
        self,
        *,
        segment: dict,
        concept: str,
        global_context: dict,
        custom_instructions: str = "",
        trigger_word: str = "",
    ) -> str:
        payload = {
            "segment": segment,
            "performance_mode": segment.get("type", ""),
            "scene_concept": concept,
            "current_visual_prompt": concept,
            "global_subject": global_context["subject"],
            "story_idea": global_context["story_idea"],
            "style": global_context["style"],
            "locations": global_context["locations"],
            "prompt_guidance": global_context.get("prompt_guidance", {}),
            "custom_instructions": custom_instructions,
            "trigger_word": trigger_word,
        }

        result = self.llm.complete_prompt(
            system_prompt=build_t2i_system_prompt(),
            prompt=json.dumps(payload, ensure_ascii=False, indent=2),
        )

        result = clean_llm_text(result)

        if trigger_word and trigger_word not in result:
            result = f"{trigger_word}, {result}"

        return result

    def build_ltx_base_prompt(
        self,
        *,
        segment: dict,
        concept: str,
        scene_details: dict,
        global_context: dict,
        custom_instructions: str = "",
    ) -> str:
        payload = build_video_payload(
            segment=segment,
            concept=concept,
            scene_details=scene_details,
            global_context=global_context,
            custom_instructions=custom_instructions,
        )

        return clean_llm_text(
            self.llm.complete_prompt(
                system_prompt=build_i2v_system_prompt(str(segment.get("type", ""))),
                prompt=json.dumps(payload, ensure_ascii=False, indent=2),
            )
        )

    def build_scene_prompts(
        self,
        *,
        stage1_segments: list[dict],
        concept_prompts: dict,
        scene_details: dict,
        global_context: dict,
        output_json_path: str | Path,
        zimage_instructions: str = "",
        ltx_instructions: str = "",
        trigger_word: str = "",
    ) -> Path:
        output = []

        for segment in stage1_segments:
            segment_id = segment["segment_id"]
            concept = concept_prompts[segment_id]
            details = scene_details.get(segment_id, {})

            zimage_prompt = self.build_zimage_prompt(
                segment=segment,
                concept=concept,
                global_context=global_context,
                custom_instructions=zimage_instructions,
                trigger_word=trigger_word,
            )

            ltx_prompt = self.build_ltx_base_prompt(
                segment=segment,
                concept=concept,
                scene_details=details,
                global_context=global_context,
                custom_instructions=ltx_instructions,
            )

            output.append({
                **segment,
                "base_concept": concept,
                "camera_motion": details.get("camera_motion", ""),
                "character_motion": details.get("character_motion", ""),
                "zimage_prompt": zimage_prompt,
                "ltx_base_prompt": ltx_prompt,
            })

        output_json_path = Path(output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return output_json_path
