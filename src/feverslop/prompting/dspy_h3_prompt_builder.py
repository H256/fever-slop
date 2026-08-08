from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def _reference(
    *,
    label: str,
    source: str | Path,
    kind: str,
    name: str,
    description: str = "",
    role: str = "general",
) -> dict[str, str]:
    source_text = str(source).replace("\\", "/")
    return {
        "label": label,
        "source": source_text,
        "kind": kind,
        "name": name,
        "description": description,
        "role": role,
    }


def _scene_references(
    segment: dict[str, Any],
    audio_paths: dict[str, Path] | None,
    reference_root: Path | None,
) -> tuple[list[dict[str, str]], list[Path]]:
    references = segment.get("references") or {}
    result: list[dict[str, str]] = []
    images: list[Path] = []

    actor_paths = references.get("actor_msr_paths") or references.get("actor_sheet_paths") or []
    actor_ids = references.get("actor_ids") or []
    for index, source in enumerate(actor_paths, start=1):
        name = str(actor_ids[index - 1]) if index <= len(actor_ids) else f"Actor {index}"
        path = Path(source)
        image_path = path if path.is_absolute() or reference_root is None else reference_root / path
        result.append(_reference(label=f"<Picture {index}>", source=path, kind="picture", name=name, role="subject"))
        if image_path.is_file():
            images.append(image_path)

    location = references.get("location_msr_path") or references.get("location_sheet_path")
    if location:
        path = Path(location)
        image_path = path if path.is_absolute() or reference_root is None else reference_root / path
        result.append(_reference(
            label=f"<Picture {len(result) + 1}>",
            source=path,
            kind="picture",
            name=str(references.get("location_id") or "Location"),
            role="environment",
        ))
        if image_path.is_file():
            images.append(image_path)

    for index, (name, source) in enumerate((audio_paths or {}).items(), start=1):
        result.append(_reference(
            label=f"<Audio {index}>",
            source=source,
            kind="audio",
            name=name,
            description="Use this synchronized stem for the scene's audio behavior.",
            role="audio_reuse",
        ))

    return result, images


class DspyH3PromptBuilder:
    """Adapter around the DSPy scene generator used by the H3 R2V pipeline."""

    def __init__(self, generator: Callable[[dict[str, Any]], Any], *, reference_root: Path | None = None):
        self.generator = generator
        self.reference_root = reference_root

    def build_h3_prompt(
        self,
        *,
        segment: dict[str, Any],
        concept: str,
        scene_details: dict[str, Any],
        global_context: dict[str, Any],
        mode: str = "ref",
        video_type: str = "music_video",
        audio_paths: dict[str, Path] | None = None,
        reference_root: Path | None = None,
    ) -> dict[str, Any]:
        references, images = _scene_references(
            segment,
            audio_paths,
            reference_root or self.reference_root,
        )
        request = {
            "mode": mode,
            "video_type": video_type,
            "duration_seconds": segment.get("duration") or segment.get("duration_seconds"),
            "user_prompt": concept,
            "notes": json.dumps({
                "scene": segment,
                "scene_details": scene_details,
                "global_context": global_context,
            }, ensure_ascii=False),
            "references": references,
            "images": images,
            "strict_fidelity": True,
        }
        try:
            generated = self.generator(request)
            prompt = getattr(generated, "rendered_prompt", None)
            if not prompt and isinstance(generated, dict):
                prompt = generated.get("rendered_prompt") or generated.get("prompt")
            if not prompt:
                raise ValueError("DSPy generator returned no rendered prompt")
        except Exception:
            prompt = concept
        return {"prompt": str(prompt).strip(), "references": references}

    def build_all_h3_prompts(
        self,
        *,
        stage1_segments: list[dict],
        concept_prompts: dict,
        scene_details: dict,
        global_context: dict,
        mode: str = "ref",
        video_type: str = "music_video",
        output_json_path: str | Path,
        artifact_store,
        audio_paths: dict[str, Path] | None = None,
        reference_root: Path | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        results = []
        total = len(stage1_segments)
        for current, segment in enumerate(stage1_segments, start=1):
            segment_id = segment["segment_id"]
            concept = concept_prompts.get(segment_id, "")
            if isinstance(concept, dict):
                concept = concept.get("concept", "")
            result = self.build_h3_prompt(
                segment=segment,
                concept=str(concept),
                scene_details=scene_details.get(segment_id, {}),
                global_context=global_context,
                mode=mode,
                video_type=video_type,
                audio_paths=audio_paths,
                reference_root=reference_root,
            )
            results.append({"segment_id": segment_id, **result})
            if progress_callback is not None:
                progress_callback(current, total)
        return artifact_store.write_json(output_json_path, results)


def build_dspy_generator(llm: Any) -> Callable[[dict[str, Any]], Any]:
    """Create the complete planner/analyzer/renderer generator from dspy_prompt_test."""
    from feverslop.prompting.dspy_h3_generator import VideoPromptGenerator

    guides = Path(__file__).with_name("guides")
    return VideoPromptGenerator(
        base_guide_path=guides / "base.md",
        reference_guide_path=guides / "reference.md",
        llm=llm,
    )