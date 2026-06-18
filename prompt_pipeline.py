from __future__ import annotations

from pathlib import Path
import json

from autoprompter.adapters.local_artifacts import JsonArtifactStore
from autoprompter.application.llm_parsing import extract_json_object
from music_video_prompt_style import (
    build_concept_mapper_system_prompt,
    build_detail_system_prompt,
    build_i2v_system_prompt,
    build_t2i_system_prompt,
    build_video_payload,
)
from autoprompter.ports.llm import LLMPort


class MusicVideoPromptPipeline:
    def __init__(self, llm: LLMPort):
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
        prompt = json.dumps(
            {
                "STORY_IDEA": story_idea,
                "SEGMENT_TIMELINE_JSON": stage1_segments,
            },
            ensure_ascii=False,
            indent=2,
        )

        response = self.llm.complete_prompt(
            system_prompt=build_concept_mapper_system_prompt(batch=False),
            prompt=prompt,
        )

        return extract_json_object(response)

    def create_scene_details(
        self,
        concept_prompts: dict,
        stage1_segments: list[dict] | None = None,
        global_context: dict | None = None,
    ) -> dict:
        details = {}
        segment_types = {
            str(segment.get("segment_id")): str(segment.get("type", ""))
            for segment in (stage1_segments or [])
        }

        for segment_id, concept in concept_prompts.items():
            segment_type = segment_types.get(str(segment_id), "")
            if isinstance(concept, dict):
                segment_type = str(concept.get("type", ""))
                concept_text = str(concept.get("concept", ""))
            else:
                concept_text = str(concept)

            camera_motion = self.llm.complete_prompt(
                system_prompt=build_detail_system_prompt("Camera Motion", segment_type=segment_type),
                prompt=json.dumps(
                    {
                        "scene_concept": concept_text,
                        "prompt_guidance": (global_context or {}).get("prompt_guidance", {}),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            ).strip()

            character_motion = self.llm.complete_prompt(
                system_prompt=build_detail_system_prompt("Character Motion", segment_type=segment_type),
                prompt=json.dumps(
                    {
                        "scene_concept": concept_text,
                        "prompt_guidance": (global_context or {}).get("prompt_guidance", {}),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
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

            t2i_prompt = self.llm.complete_prompt(
                system_prompt=build_t2i_system_prompt(),
                prompt=json.dumps(
                    {
                        "segment": segment,
                        "performance_mode": segment.get("type", ""),
                        "scene_concept": concept,
                        "current_visual_prompt": concept,
                        "global_subject": global_context["subject"],
                        "story_idea": global_context["story_idea"],
                        "style": global_context["style"],
                        "locations": global_context["locations"],
                        "prompt_guidance": global_context.get("prompt_guidance", {}),
                        "custom_instructions": "",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            ).strip()

            prompt_payload = json.dumps(
                build_video_payload(
                    segment=segment,
                    concept=concept,
                    scene_details=details,
                    global_context=global_context,
                    t2i_prompt=t2i_prompt,
                ),
                ensure_ascii=False,
                indent=2,
            )

            final_prompt = self.llm.complete_prompt(
                system_prompt=build_i2v_system_prompt(str(segment.get("type", ""))),
                prompt=prompt_payload,
            ).strip()

            result.append({
                **segment,
                "base_concept": concept,
                "camera_motion": details["camera_motion"],
                "character_motion": details["character_motion"],
                "zimage_prompt": t2i_prompt,
                "t2i_prompt": t2i_prompt,
                "ltx_base_prompt": t2i_prompt,
                "i2v_prompt_from_t2i": final_prompt,
                "final_prompt": final_prompt,
            })

        return result

    @staticmethod
    def save_json(path: str | Path, data) -> Path:
        return JsonArtifactStore().write_json(path, data)
