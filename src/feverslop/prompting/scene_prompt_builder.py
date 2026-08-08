from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Callable

from feverslop.prompting.music_video_prompt_style import (
    build_i2v_system_prompt,
    build_t2i_system_prompt,
    build_video_payload,
)
from feverslop.ports.artifacts import ArtifactStore
from feverslop.ports.llm import LLMPort
from feverslop.domain.scene_cast import resolve_scene_cast, scene_cast_to_prompt_payload


def clean_llm_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:text)?", "", text.strip(), flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text.strip()).strip()
    return text.strip()


def normalize_scene_references(references: dict, global_context: dict) -> dict:
    actors = [
        str(actor.get("id", "")).strip()
        for actor in global_context.get("actors", [])
        if isinstance(actor, dict) and str(actor.get("id", "")).strip()
    ]
    locations = [
        str(location.get("id", "")).strip()
        for location in global_context.get("structured_locations", [])
        if isinstance(location, dict) and str(location.get("id", "")).strip()
    ]
    subject_mode = str(global_context.get("subject_mode", "multi") or "multi").strip().lower()
    max_scene_actors = int(global_context.get("max_scene_actors", 1 if subject_mode == "single" else 4) or 4)
    max_scene_actors = max(1, min(4, max_scene_actors))

    output = dict(references or {})
    if actors:
        if subject_mode == "single":
            output["actor_ids"] = [actors[0]]
        else:
            selected = [
                str(actor_id).strip()
                for actor_id in output.get("actor_ids", [])
                if str(actor_id).strip() in actors
            ]
            output["actor_ids"] = (selected or [actors[0]])[:max_scene_actors]

    if locations:
        location_id = str(output.get("location_id", "")).strip()
        if location_id not in locations:
            location_id = locations[0]
        output["location_id"] = location_id

    return output


class ScenePromptBuilder:
    """
    Builds model-specific prompts per scene.

    Important design:
    - Subject is injected into EVERY scene-generation call.
    - Z-Image prompts are still-image keyframe prompts.
    - LTX prompts are video prompts and may contain motion.
    """

    def __init__(self, llm: LLMPort):
        self.llm = llm

    def build_zimage_prompt(
        self,
        *,
        segment: dict,
        concept: str,
        global_context: dict,
        scene_cast: dict | None = None,
        custom_instructions: str = "",
        trigger_word: str = "",
    ) -> str:
        payload = {
            "segment": segment,
            "performance_mode": segment.get("type", ""),
            "scene_concept": concept,
            "current_visual_prompt": concept,
            "scene_cast": scene_cast or {},
            "global_subject": global_context["subject"],
            "story_idea": global_context["story_idea"],
            "style": global_context["style"],
            "locations": global_context["locations"],
            "location_constraint": global_context.get("location_constraint", ""),
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
        t2i_prompt: str = "",
        custom_instructions: str = "",
    ) -> str:
        return self.build_i2v_prompt_from_t2i(
            segment=segment,
            concept=concept,
            scene_details=scene_details,
            global_context=global_context,
            t2i_prompt=t2i_prompt,
            custom_instructions=custom_instructions,
        )

    def build_i2v_prompt_from_t2i(
        self,
        *,
        segment: dict,
        concept: str,
        scene_details: dict,
        global_context: dict,
        scene_cast: dict | None = None,
        t2i_prompt: str = "",
        custom_instructions: str = "",
    ) -> str:
        payload = build_video_payload(
            segment=segment,
            concept=concept,
            scene_details=scene_details,
            global_context=global_context,
            scene_cast=scene_cast,
            t2i_prompt=t2i_prompt,
            custom_instructions=custom_instructions,
        )

        return clean_llm_text(
            self.llm.complete_prompt(
                system_prompt=build_i2v_system_prompt(
                    str(segment.get("type", "")),
                    silent_mode=bool(global_context.get("silent_mode", False)),
                ),
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
        artifact_store: ArtifactStore,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        output = []
        total = len(stage1_segments)

        for current, segment in enumerate(stage1_segments, start=1):
            segment_id = segment["segment_id"]
            concept = concept_prompts[segment_id]
            references = {}
            if isinstance(concept, dict):
                references = normalize_scene_references(
                    dict(concept.get("references") or {}),
                    global_context,
                )
                concept = str(concept.get("concept", ""))
            cast = resolve_scene_cast(
                selected_actor_ids=references.get("actor_ids") or [],
                available_actors=global_context.get("actors") or [],
                subject_mode=str(global_context.get("subject_mode") or "multi"),
                max_scene_actors=int(global_context.get("max_scene_actors") or 4),
            )
            scene_cast = scene_cast_to_prompt_payload(cast)
            details = scene_details.get(segment_id, {})

            t2i_prompt = self.build_zimage_prompt(
                segment=segment,
                concept=concept,
                global_context=global_context,
                scene_cast=scene_cast,
                custom_instructions=zimage_instructions,
                trigger_word=trigger_word,
            )

            i2v_prompt_from_t2i = self.build_i2v_prompt_from_t2i(
                segment=segment,
                concept=concept,
                scene_details=details,
                global_context=global_context,
                scene_cast=scene_cast,
                t2i_prompt=t2i_prompt,
                custom_instructions=ltx_instructions,
            )

            scene_output = {
                **segment,
                "base_concept": concept,
                "silent_mode": bool(global_context.get("silent_mode", False)),
                "camera_motion": details.get("camera_motion", ""),
                "character_motion": details.get("character_motion", ""),
                "zimage_prompt": t2i_prompt,
                "t2i_prompt": t2i_prompt,
                "ltx_base_prompt": t2i_prompt,
                "i2v_prompt_from_t2i": i2v_prompt_from_t2i,
                "original_style_i2v_prompt": i2v_prompt_from_t2i,
            }
            if references:
                scene_output["references"] = references
            output.append(scene_output)
            if progress_callback is not None:
                progress_callback(current, total)

        return artifact_store.write_json(output_json_path, output)
