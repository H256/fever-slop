from __future__ import annotations

from pathlib import Path
import json
from typing import Callable

from feverslop.domain.llm_parsing import extract_json_object
from feverslop.prompting.minimax_h3_prompt_style import build_h3_video_system_prompt
from feverslop.ports.artifacts import ArtifactStore
from feverslop.ports.llm import LLMPort


def _build_references_from_segment(segment: dict) -> dict | None:
    """Extract reference labels from segment for R2V system prompt.

    Returns dict with:
      - subjects: list[{"name": ..., "description": ..., "type": "actor"|"location"|"audio"|"video"}]
        Actors/locations sourced from ref_items; audio/video sourced from paths.
      - speaker_ids: {"actor_name": "S1", ...} mapping actor names to speaker IDs
      - image: list[{"name": ..., "path": ..., "description": ...}]
      - video: list[{"name": ..., "path": ...}]
      - audio: list[{"name": ..., "path": ...}]
    Or None if no refs present.
    """
    refs = segment.get("references", {})
    if not refs:
        return None
    image_paths = refs.get("reference_image_paths", [])
    video_paths = refs.get("reference_video_paths", [])
    audio_paths = refs.get("reference_audio_paths", [])
    if not image_paths and not video_paths and not audio_paths:
        return None

    ref_items = segment.get("ref_items", [])

    # Build subjects from ref_items of type actor/location
    subjects: list[dict] = []
    speaker_ids: dict[str, str] = {}
    speaker_counter = 0
    for ref_item in ref_items:
        if ref_item.get("type") in ("actor", "location"):
            name = ref_item.get("name", f"Subject {len(subjects) + 1}")
            desc = ref_item.get("visual_description", "")
            subjects.append({"name": name, "description": desc, "type": ref_item.get("type", "actor")})
            if ref_item.get("type") == "actor":
                speaker_counter += 1
                speaker_ids[name] = f"S{speaker_counter}"

    # Build image refs: pair with image_paths by order, matching existing logic
    image_data: list[dict] = []
    for i, path in enumerate(image_paths):
        basename = str(path).split("/")[-1].rsplit(".", 1)[0]
        if i < len(ref_items) and ref_items[i].get("type") in ("actor", "location"):
            name = ref_items[i].get("name", basename)
            desc = ref_items[i].get("visual_description", "")
        else:
            name = basename
            desc = ""
        image_data.append({"name": name, "path": str(path), "description": desc})

    # Video refs
    video_data: list[dict] = [{"name": str(p).split("/")[-1].rsplit(".", 1)[0], "path": str(p)} for p in video_paths]
    for vid in video_data:
        subjects.append({"name": vid["name"], "description": "", "type": "video"})

    # Audio refs
    audio_data: list[dict] = [{"name": str(p).split("/")[-1].rsplit(".", 1)[0], "path": str(p)} for p in audio_paths]
    for aud in audio_data:
        subjects.append({"name": aud["name"], "description": "", "type": "audio"})

    if not image_data and not video_data and not audio_data:
        return None

    result: dict = {
        "subjects": subjects,
        "image": image_data,
        "video": video_data,
        "audio": audio_data,
    }
    if speaker_ids:
        result["speaker_ids"] = speaker_ids
    return result


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
        segment = dict(segment)
        if silent_mode:
            segment_references = dict(segment.get("references") or {})
            vocal_path = str(
                ((segment.get("stem_audio") or {}).get("paths") or {}).get("vocals") or ""
            )
            if vocal_path:
                segment_references["reference_audio_paths"] = [
                    path
                    for path in segment_references.get("reference_audio_paths", [])
                    if str(path) != vocal_path
                ]
            segment["references"] = segment_references
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
        audio_paths: dict[str, Path] | None = None,
        reference_root: Path | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        status_callback: Callable[[int, int, str], None] | None = None,
    ) -> Path:
        """Per-scene batch generation. Returns written file path."""
        results = []
        total = len(stage1_segments)
        for current, segment in enumerate(stage1_segments, start=1):
            segment_id = segment["segment_id"]
            if status_callback is not None:
                status_callback(current, total, "started")
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
            if progress_callback is not None:
                progress_callback(current, total)
            if status_callback is not None:
                status_callback(current, total, "completed")

        return artifact_store.write_json(output_json_path, results)
