from __future__ import annotations

from pathlib import Path
import json
import re

from llm_client import LocalOpenAIClient


def extract_json_object(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text.strip()).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in LLM response:\n{text}")

    return json.loads(text[start:end + 1])


class MusicVideoPromptPipeline:
    def __init__(self, llm: LocalOpenAIClient):
        self.llm = llm

    def create_story_idea(
        self,
        lyrics: str,
        notes: str = "",
    ) -> str:
        system_prompt = """
You turn song lyrics and optional user notes into a short story idea/concept.

Input:
- Lyrics
- Optional notes such as style, genre, mood, setting, characters, themes, or constraints

Task:
Create one concise short story idea inspired by the lyrics and notes.

Rules:
- Keep the final concept under 1000 characters.
- Do not quote or reuse long lyric phrases.
- Capture the emotional core, imagery, conflict, or theme of the lyrics.
- If notes are provided, follow them.
- If notes conflict with the lyrics, blend them creatively.
- Output only the story concept.
- No explanations, titles, bullet points, or extra text.

Style:
- Clear, vivid, and specific.
- Prefer cinematic story hooks.
- Avoid vague concepts like "a person learns about love."
- Make it feel like a story premise, not a summary of the song.
""".strip()

        return self.llm.complete_prompt(
            system_prompt=system_prompt,
            prompt=f"LYRICS:\n{lyrics}\n\nNOTES:\n{notes}",
        ).strip()

    def create_style_block(
        self,
        lyrics: str,
        notes: str = "",
    ) -> str:
        system_prompt = """
send back ONLY this 3-part block:

STYLE / THEME
1 short sentence describing the overall feeling, tone, and visual direction.

COLOR PALETTE
1 short line describing the main colors and accent colors. Never fade into dark colors.

LIGHTING / MOOD
1 short line describing brightness, contrast, and shadows.

Rules:
Use simple, everyday words.
Keep the full output under 1000 characters.
Do not include camera, lens, framing, composition, or extra detail sections.
Avoid metaphors, symbolism, poetic language, and extra explanation.
Output only the block.
""".strip()

        return self.llm.complete_prompt(
            system_prompt=system_prompt,
            prompt=f"LYRICS:\n{lyrics}\n\nNOTES:\n{notes}",
        ).strip()

    def create_subject_and_locations(
        self,
        story_idea: str,
        notes: str = "",
    ) -> dict:
        system_prompt = """
You extract one consistent subject and a short list of usable physical locations for a music video.

Return ONLY valid JSON in this exact shape:
{
  "subject": "a [gender/person], with [hair color], wearing [outfit]",
  "locations": [
    "short physical location 1",
    "short physical location 2"
  ]
}

Rules for subject:
- Create one simple subject only.
- Infer gender only if clearly implied. If unclear, use: person.
- If hair color is not mentioned, invent a reasonable default that fits the tone.
- Only include gender/person, hair color, and outfit.

Rules for locations:
- List only physical environments where a person could realistically be standing.
- Avoid aerial, drone, satellite, or far landscape shots.
- No camera directions.
- No emotional explanations.
""".strip()

        response = self.llm.complete_prompt(
            system_prompt=system_prompt,
            prompt=f"STORY_IDEA:\n{story_idea}\n\nNOTES:\n{notes}",
        )

        return extract_json_object(response)

    def create_concept_prompts(
        self,
        stage1_segments: list[dict],
        story_idea: str,
    ) -> dict:
        system_prompt = """
You are a music-video visual concept mapper.

INPUTS:
1. SEGMENT_TIMELINE_JSON: timed song sections. Each has segment_id, start, end, duration, type, and optional lyrics.
2. STORY_IDEA: the overall visual narrative goal.

TASK:
Create exactly one concise visual concept for each segment.

Rules:
- For "vocals": reflect the lyrics and advance the story. The character may sing or lip-sync.
- For "instrumental": advance the story visually without referencing sung words. Do not invent lyrics. Do not say the character is singing.
- For "mixed": combine lyrical meaning with instrumental mood.
- Maintain character, location, style, and narrative continuity.
- Describe only visible subjects, actions, settings, mood, and camera-relevant visual elements.
- No technical parameters, no frame numbers, no markdown, no comments.

OUTPUT:
Return ONLY valid JSON.
Each key must exactly match the segment_id.
Each value must be one concise visual concept string.
""".strip()

        prompt = json.dumps(
            {
                "STORY_IDEA": story_idea,
                "SEGMENT_TIMELINE_JSON": stage1_segments,
            },
            ensure_ascii=False,
            indent=2,
        )

        response = self.llm.complete_prompt(
            system_prompt=system_prompt,
            prompt=prompt,
        )

        return extract_json_object(response)

    def create_scene_details(
        self,
        concept_prompts: dict,
    ) -> dict:
        details = {}

        for segment_id, concept in concept_prompts.items():
            camera_motion = self.llm.complete_prompt(
                system_prompt="""
You create one cinematic camera motion for a music video scene.
Return only one short phrase. No explanations.
Examples: slow push-in, low-angle tracking shot, handheld orbit, gentle crane down.
""".strip(),
                prompt=f"SCENE CONCEPT:\n{concept}",
            ).strip()

            character_motion = self.llm.complete_prompt(
                system_prompt="""
You create one visible character motion for a music video scene.
Return only one short phrase. No explanations.
Examples: walks slowly forward, turns toward the light, raises one hand, stands still breathing.
""".strip(),
                prompt=f"SCENE CONCEPT:\n{concept}",
            ).strip()

            details[segment_id] = {
                "camera_motion": camera_motion,
                "character_motion": character_motion,
            }

        return details

    def create_final_scene_prompts(
        self,
        stage1_segments: list[dict],
        concept_prompts: dict,
        scene_details: dict,
        global_context: dict,
    ) -> list[dict]:
        result = []

        for segment in stage1_segments:
            segment_id = segment["segment_id"]
            concept = concept_prompts[segment_id]
            details = scene_details[segment_id]

            system_prompt = """
You create one final cinematic prompt for an AI video/image generation workflow.

Inputs:
- subject
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
- Preserve the scene timing fields from the input; do not mention timing in the prompt.
- If segment type is instrumental: the character must not sing and must have no lip-sync.
- If segment type is vocals: the character may sing or lip-sync.
- If segment type is mixed: describe a continuous shot that can support both instrumental and singing moments.
- Use visible, cinematic language.
- Return only one polished prompt. No JSON. No markdown.
""".strip()

            prompt_payload = json.dumps(
                {
                    "subject": global_context["subject"],
                    "story_idea": global_context["story_idea"],
                    "style": global_context["style"],
                    "locations": global_context["locations"],
                    "segment": segment,
                    "scene_concept": concept,
                    "camera_motion": details["camera_motion"],
                    "character_motion": details["character_motion"],
                },
                ensure_ascii=False,
                indent=2,
            )

            final_prompt = self.llm.complete_prompt(
                system_prompt=system_prompt,
                prompt=prompt_payload,
            ).strip()

            result.append({
                **segment,
                "base_concept": concept,
                "camera_motion": details["camera_motion"],
                "character_motion": details["character_motion"],
                "final_prompt": final_prompt,
            })

        return result

    @staticmethod
    def save_json(path: str | Path, data) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path