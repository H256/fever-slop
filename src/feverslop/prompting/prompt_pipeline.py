from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from feverslop.domain.llm_parsing import extract_json_object
from feverslop.domain.scene_cast import resolve_scene_cast, scene_cast_to_prompt_payload
from feverslop.ports.artifacts import ArtifactStore
from feverslop.ports.llm import LLMPort
from feverslop.prompting.music_video_modules import MusicVideoPromptModules
from feverslop.prompting.music_video_prompt_style import (
    build_detail_system_prompt,
    build_i2v_system_prompt,
    build_t2i_system_prompt,
    build_video_payload,
)


class MusicVideoPromptPipeline:
    def __init__(self, llm: LLMPort, *, prompt_modules: MusicVideoPromptModules | None = None):
        self.llm = llm
        self.prompt_modules = prompt_modules or MusicVideoPromptModules(llm)

    def create_story_idea(
        self,
        lyrics: str,
        notes: str = "",
    ) -> str:
        return self.prompt_modules.story_idea(lyrics, notes)

    def create_style_block(
        self,
        lyrics: str,
        notes: str = "",
    ) -> str:
        return self.prompt_modules.style_block(lyrics, notes)

    def create_subject_and_locations(
        self,
        story_idea: str,
        notes: str = "",
    ) -> dict:
        response = self.prompt_modules.subject_locations(story_idea, notes)
        if not isinstance(response, str):
            response = response.model_dump_json() if hasattr(response, "model_dump_json") else json.dumps(response)

        data = extract_json_object(response)
        for location in data.get("locations") or []:
            if not isinstance(location, dict):
                continue
            prompt = str(location.get("image_prompt") or "")
            prompt = re.sub(r"(?i)\b(?:cinematic\s+)?environment reference sheet(?:\s+for)?\b", "", prompt)
            prompt = re.sub(r"(?i)\breference sheet\b", "", prompt)
            location["image_prompt"] = " ".join(prompt.split()).strip(" .,-")
        return data

    def create_concept_prompts(
        self,
        stage1_segments: list[dict],
        story_idea: str,
        global_context: dict | None = None,
        notes: str = "",
    ) -> dict:
        response = self.prompt_modules.concepts(
            {"STORY_IDEA": story_idea, "GLOBAL_CONTEXT": global_context or {}, "NOTES": notes,
             "SEGMENT_TIMELINE_JSON": stage1_segments},
            silent_mode=bool((global_context or {}).get("silent_mode", False)),
        )
        return response if isinstance(response, dict) else extract_json_object(str(response))

    def create_scene_details(
        self,
        concept_prompts: dict,
        stage1_segments: list[dict] | None = None,
        global_context: dict | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict:
        details = {}
        segment_types = {
            str(segment.get("segment_id")): str(segment.get("type", ""))
            for segment in (stage1_segments or [])
        }

        total = len(concept_prompts)
        for current, (segment_id, concept) in enumerate(concept_prompts.items(), start=1):
            segment_type = segment_types.get(str(segment_id), "")
            references = {}
            if isinstance(concept, dict):
                segment_type = str(concept.get("type", ""))
                concept_text = str(concept.get("concept", ""))
                references = dict(concept.get("references") or {})
            else:
                concept_text = str(concept)
            context = global_context or {}
            scene_cast = scene_cast_to_prompt_payload(resolve_scene_cast(
                selected_actor_ids=references.get("actor_ids") or [],
                available_actors=context.get("actors") or [],
                subject_mode=str(references.get("subject_mode") or context.get("subject_mode") or "multi"),
                max_scene_actors=int(context.get("max_scene_actors") or 4),
                scene_number=references.get("scene") or segment_id,
            ))

            detail_payload = {
                "scene_concept": concept_text, "scene_cast": scene_cast,
                "prompt_guidance": (global_context or {}).get("prompt_guidance", {}),
            }
            camera_motion = self.prompt_modules.detail(
                "Camera Motion", detail_payload,
                build_detail_system_prompt("Camera Motion", segment_type=segment_type,
                    silent_mode=bool((global_context or {}).get("silent_mode", False))),
            )
            character_motion = self.prompt_modules.detail(
                "Character Motion", detail_payload,
                build_detail_system_prompt("Character Motion", segment_type=segment_type,
                    silent_mode=bool((global_context or {}).get("silent_mode", False))),
            )
            spatial_relations = self.prompt_modules.detail(
                "Spatial Relations", detail_payload,
                build_detail_system_prompt("Spatial Relations", segment_type=segment_type,
                    silent_mode=bool((global_context or {}).get("silent_mode", False))),
            )

            details[segment_id] = {
                "camera_motion": camera_motion,
                "character_motion": character_motion,
                "spatial_relations": spatial_relations,
            }
            if progress_callback is not None:
                progress_callback(current, total)

        return details

    def create_final_scene_prompts(
        self,
        stage1_segments: list[dict],
        concept_prompts: dict,
        scene_details: dict,
        global_context: dict,
    ) -> list[dict]:
        segment_ids = {seg["segment_id"] for seg in stage1_segments}
        missing_in_concepts = segment_ids - set(concept_prompts.keys())
        missing_in_details = segment_ids - set(scene_details.keys())
        if missing_in_concepts or missing_in_details:
            parts = []
            if missing_in_concepts:
                parts.append(f"concept_prompts: {sorted(missing_in_concepts)}")
            if missing_in_details:
                parts.append(f"scene_details: {sorted(missing_in_details)}")
            raise ValueError(
                f"Segment IDs from stage1_segments missing in upstream results: {', '.join(parts)}",
            )
        result = []

        for segment in stage1_segments:
            segment_id = segment["segment_id"]
            concept = concept_prompts[segment_id]
            details = scene_details[segment_id]

            t2i_payload = {
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
                    }
            t2i_prompt = self.prompt_modules.t2i(t2i_payload, build_t2i_system_prompt())

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

            final_prompt = self.prompt_modules.i2v(
                json.loads(prompt_payload),
                build_i2v_system_prompt(str(segment.get("type", "")),
                    silent_mode=bool(global_context.get("silent_mode", False))),
                performance_policy=str(build_video_payload(
                    segment=segment, concept=concept, scene_details=details,
                    global_context=global_context, t2i_prompt=t2i_prompt,
                )["performance_policy"]),
            )

            result.append({
                **segment,
                "base_concept": concept,
                "camera_motion": details["camera_motion"],
                "character_motion": details["character_motion"],
                "spatial_relations": details.get("spatial_relations", ""),
                "zimage_prompt": t2i_prompt,
                "t2i_prompt": t2i_prompt,
                "ltx_base_prompt": t2i_prompt,
                "i2v_prompt_from_t2i": final_prompt,
                "final_prompt": final_prompt,
            })

        return result

    @staticmethod
    def save_json(path: str | Path, data, *, artifact_store: ArtifactStore) -> Path:
        return artifact_store.write_json(path, data)
