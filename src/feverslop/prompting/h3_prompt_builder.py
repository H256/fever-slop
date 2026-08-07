from __future__ import annotations

from pathlib import Path
import json

from feverslop.domain.llm_parsing import extract_json_object
from feverslop.prompting.minimax_h3_prompt_style import build_h3_video_system_prompt
from feverslop.ports.artifacts import ArtifactStore
from feverslop.ports.llm import LLMPort


def _build_references_from_segment(segment: dict) -> list[dict] | None:
    """Extract reference labels from segment for R2V system prompt.

    Returns list of {"label": ..., "type": "image"|"video"|"audio"} dicts.
    """
    refs = segment.get("references", {})
    if not refs:
        return None
    image_paths = refs.get("reference_image_paths", [])
    video_paths = refs.get("reference_video_paths", [])
    audio_paths = refs.get("reference_audio_paths", [])
    if not image_paths and not video_paths and not audio_paths:
        return None
    result: list[dict] = []
    ref_items = segment.get("ref_items", [])
    image_labels = []
    for ref_item in ref_items:
        if ref_item.get("type") in ("actor", "location"):
            label = ref_item.get("name", "")
            if label:
                image_labels.append(label)
    if len(image_labels) < len(image_paths):
        for p in image_paths[len(image_labels):]:
            image_labels.append(str(p).split("/")[-1].rsplit(".", 1)[0])
    for label in image_labels:
        result.append({"label": label, "type": "image"})
    for p in video_paths:
        label = str(p).split("/")[-1].rsplit(".", 1)[0]
        result.append({"label": label, "type": "video"})
    for p in audio_paths:
        label = str(p).split("/")[-1].rsplit(".", 1)[0]
        result.append({"label": label, "type": "audio"})
    return result if result else None


class H3PromptBuilder:
    """Builds H3-structured prompts per scene, mirroring ScenePromptBuilder pattern."""

    def __init__(self, llm: LLMPort):
        self.llm = llm

    def build_h3_prompt(
        self,
        *,
        segment: dict,
        concept: str,
        scene_details: dict,
        global_context: dict,
        mode: str = "base",
        video_type: str = "music_video",
    ) -> dict:
        """Generate H3-structured prompt for one scene.

        Returns dict with H3 fields plus a merged `prompt` string key.
        """
        silent_mode = bool(global_context.get("silent_mode", False))
        has_audio_refs = bool(segment.get("references", {}).get("reference_audio_paths"))

        # Build references dict from segment for ref mode
        references = _build_references_from_segment(segment)
        system_prompt = build_h3_video_system_prompt(
            mode=mode,
            video_type=video_type,
            silent_mode=silent_mode,
            references=references,
        )

        payload = {
            "segment": segment,
            "performance_mode": segment.get("type", ""),
            "scene_concept": concept,
            "camera_motion": json.dumps(scene_details, ensure_ascii=False, indent=2) if isinstance(scene_details, dict) else scene_details,
            "character_motion": "",
            "global_subject": global_context.get("subject", ""),
            "story_idea": global_context.get("story_idea", ""),
            "style": global_context.get("style", ""),
            "locations": global_context.get("locations", []),
            "location_constraint": global_context.get("location_constraint", ""),
            "silent_mode": silent_mode,
            "has_audio_refs": has_audio_refs,
        }

        try:
            response = self.llm.complete_prompt(
                system_prompt=system_prompt,
                prompt=json.dumps(payload, ensure_ascii=False, indent=2),
            )
            structured: dict = extract_json_object(response)
        except Exception:
            # Fallback: return raw text in prompt key
            structured = {"prompt": str(response).strip()}

        # Build merged prompt string if not present
        if "prompt" not in structured:
            merged_parts: list[str] = []
            # For ref mode
            if mode == "ref":
                for field in ("subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape", "non_diegetic_music"):
                    val = structured.get(field, "")
                    if val:
                        merged_parts.append(f"{field}: {val}")
            else:
                for field in ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"):
                    val = structured.get(field, "")
                    if val:
                        merged_parts.append(f"{field}: {val}")
            structured["prompt"] = "\n".join(merged_parts)

        return structured

    def build_all_h3_prompts(
        self,
        *,
        stage1_segments: list[dict],
        concept_prompts: dict,
        scene_details: dict,
        global_context: dict,
        mode: str = "base",
        video_type: str = "music_video",
        output_json_path: str | Path,
        artifact_store: ArtifactStore,
    ) -> Path:
        """Per-scene batch generation. Returns written file path."""
        results = []
        for segment in stage1_segments:
            segment_id = segment["segment_id"]
            concept = concept_prompts.get(segment_id, {})
            if isinstance(concept, dict):
                concept_text = str(concept.get("concept", ""))
            else:
                concept_text = str(concept)
            details = scene_details.get(segment_id, {})

            h3_data = self.build_h3_prompt(
                segment=segment,
                concept=concept_text,
                scene_details=details,
                global_context=global_context,
                mode=mode,
                video_type=video_type,
            )
            entry = {"segment_id": segment_id, **h3_data}
            results.append(entry)

        return artifact_store.write_json(output_json_path, results)
