from __future__ import annotations

from pathlib import Path

from feverslop.prompting.dspy_h3_prompt_builder import (
    DspyH3PromptBuilder,
    build_dspy_generator,
)


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


def build_references_from_segment(segment: dict) -> dict | None:
    """Build the normalized H3 reference payload for a segment."""
    return _build_references_from_segment(segment)


class H3PromptBuilder(DspyH3PromptBuilder):
    """Compatibility facade backed by the canonical DSPy H3 builder."""

    def __init__(self, llm, *, reference_root: Path | None = None, allow_fallback: bool = True):
        super().__init__(
            build_dspy_generator(llm),
            reference_root=reference_root,
            allow_fallback=allow_fallback,
        )
