from __future__ import annotations

from pathlib import Path
import json
import re

from llm_client import LocalOpenAIClient


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
        system_prompt = """
You are a text-to-image prompt generator that creates one highly detailed prompt for image generation.

After the user provides details, generate one single prompt based on their input.
If details are missing, creatively fill in the missing visual elements while keeping the prompt coherent.

Your output must be only the final prompt text. Do not output a list, explanation, table, JSON, or markdown.

Critical rules:
- Generate a still image prompt, not a video prompt.
- Describe one frozen cinematic moment.
- Do NOT include camera movement.
- Do NOT include lip sync.
- Do NOT say the character is singing.
- Do NOT describe actions that require multiple frames.
- Preserve the given subject exactly and keep it visually consistent.
- If the scene is instrumental, the subject must not sing and must not have mouth movement.
- If the scene is vocal/mixed, the keyframe may show the subject in an expressive performance pose, but still avoid "singing", "lip sync", or moving lips.
- Avoid quoting lyrics.

Required output structure:
"[Color Style + Mood] photograph of [Subject description]. [Clothing and appearance details]. The scene takes place in [environment description]. Camera is [camera angle / framing]. The weather is [weather] during [time of day]. Additional cinematic details: [extra visual elements]."

Prompt creation guidelines:
- Color Style: Natural, Matte, HDR, Cinematic, Vintage, Grunge, B&W, Split Tone, High Contrast, etc.
- Mood: Bright, Dark, Epic, Dramatic, Cinematic, Peaceful, Mysterious, Somber, Mythical, etc.
- Subject: clearly describe the main subject, clothing, hairstyle, hair color, physical features, and pose.
- Environment: physical setting that complements the story.
- Camera Angle: low angle, eye level, aerial view, close-up, medium shot, wide shot.
- Weather and time of day: specific atmospheric conditions.
- Cinematic details: fog, dramatic lighting, reflections, depth of field, particles, rim light.
""".strip()

        payload = {
            "segment": segment,
            "scene_concept": concept,
            "global_subject": global_context["subject"],
            "story_idea": global_context["story_idea"],
            "style": global_context["style"],
            "locations": global_context["locations"],
            "custom_instructions": custom_instructions,
            "trigger_word": trigger_word,
        }

        result = self.llm.complete_prompt(
            system_prompt=system_prompt,
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
        system_prompt = """
You are an AI video prompt generator.

Create one final cinematic video prompt for a single scene.

Inputs include:
- exact subject
- story idea
- style block
- locations
- scene concept
- camera motion
- character motion
- segment type: vocals, instrumental, or mixed
- optional lyrics

Rules:
- Preserve the subject exactly.
- The same subject must appear consistently across scenes.
- Describe a video shot, so camera motion, environmental motion, and character motion are allowed.
- If segment type is instrumental: the subject must not sing and must not have lip movement.
- If segment type is vocals: the subject may perform vocally, but do not quote long lyrics.
- If segment type is mixed: describe a continuous shot that can support both silent and singing intervals.
- Use visible cinematic language.
- Do not mention frame numbers.
- Do not mention JSON fields.
- Output only one polished prompt. No markdown. No explanations.
""".strip()

        payload = {
            "segment": segment,
            "scene_concept": concept,
            "camera_motion": scene_details.get("camera_motion", ""),
            "character_motion": scene_details.get("character_motion", ""),
            "global_subject": global_context["subject"],
            "story_idea": global_context["story_idea"],
            "style": global_context["style"],
            "locations": global_context["locations"],
            "custom_instructions": custom_instructions,
        }

        return clean_llm_text(
            self.llm.complete_prompt(
                system_prompt=system_prompt,
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
