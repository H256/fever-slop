from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Callable

from feverslop.domain.llm_parsing import extract_json_object
from feverslop.prompting.music_video_prompt_style import build_concept_mapper_system_prompt
from feverslop.ports.artifacts import ArtifactStore
from feverslop.ports.llm import LLMPort


def chunked(items: list[Any], size: int):
    for start in range(0, len(items), size):
        yield start, items[start:start + size]


class ConceptPromptBatcher:
    """
    Robust concept-prompt generation for many music-video scenes.

    Why:
    Large single-call JSON generation often drops late keys like segment_040.
    This class generates concepts in smaller batches and validates that every
    stage1 segment receives exactly one concept.

    Continuity strategy:
    Every batch receives:
    - full story idea
    - global context
    - project steering
    - previous generated concepts summary
    - previous few concepts verbatim

    This keeps scene progression continuous while reducing JSON failure risk.
    """

    def __init__(
        self,
        llm: LLMPort,
        batch_size: int = 10,
        max_previous_concepts: int = 6,
        request_timeout_seconds: float | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        self.llm = llm
        self.batch_size = batch_size
        self.max_previous_concepts = max_previous_concepts
        self.request_timeout_seconds = request_timeout_seconds
        self.progress_callback = progress_callback

    def create_concept_prompts_batched(
        self,
        *,
        stage1_segments: list[dict],
        story_idea: str,
        global_context: dict,
        notes: str = "",
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict:
        all_results: dict[str, str] = {}
        previous_summary = ""
        batches = list(chunked(stage1_segments, self.batch_size))
        report = progress_callback or self.progress_callback

        for batch_number, (batch_start, batch) in enumerate(batches, start=1):
            batch_label = f"Concept batch {batch_number}/{len(batches)}"
            self._report(
                f"{batch_label}: generating scenes "
                f"{batch_start + 1}-{batch_start + len(batch)}",
                report,
            )
            batch_result = self._generate_batch(
                batch_index=batch_start // self.batch_size + 1,
                batch=batch,
                story_idea=story_idea,
                global_context=global_context,
                notes=notes,
                previous_concepts=self._last_concepts(all_results),
                previous_summary=previous_summary,
            )
            self._report(f"{batch_label}: response received, validating keys", report)

            expected_ids = [seg["segment_id"] for seg in batch]
            batch_result = self._repair_missing_or_extra_keys(
                expected_ids=expected_ids,
                result=batch_result,
                batch=batch,
                story_idea=story_idea,
                global_context=global_context,
                notes=notes,
                previous_concepts=self._last_concepts(all_results),
                previous_summary=previous_summary,
                progress_callback=report,
            )

            all_results.update(batch_result)

            previous_summary = self._summarize_progress(
                story_idea=story_idea,
                global_context=global_context,
                concepts=all_results,
            )
            self._report(f"{batch_label}: complete ({len(all_results)} scenes total)", report)

        missing = [
            seg["segment_id"]
            for seg in stage1_segments
            if seg["segment_id"] not in all_results
        ]

        if missing:
            raise ValueError(f"Missing concept prompts after batched generation: {missing}")

        # Preserve stage1 order in output JSON.
        return {
            seg["segment_id"]: all_results[seg["segment_id"]]
            for seg in stage1_segments
        }

    def _report(self, message: str, callback: Callable[[str], None] | None = None) -> None:
        callback = callback or self.progress_callback
        if callback is not None:
            callback(message)

    def _generate_batch(
        self,
        *,
        batch_index: int,
        batch: list[dict],
        story_idea: str,
        global_context: dict,
        notes: str,
        previous_concepts: dict,
        previous_summary: str,
    ) -> dict:
        payload = {
            "BATCH_INDEX": batch_index,
            "STORY_IDEA": story_idea,
            "GLOBAL_CONTEXT": global_context,
            "NOTES": notes,
            "PREVIOUS_PROGRESS_SUMMARY": previous_summary,
            "PREVIOUS_CONCEPTS": previous_concepts,
            "CURRENT_BATCH_SEGMENTS": batch,
        }

        response = self.llm.complete_prompt(
            system_prompt=build_concept_mapper_system_prompt(
                batch=True,
                silent_mode=bool(global_context.get("silent_mode", False)),
            ),
            prompt=json.dumps(payload, ensure_ascii=False, indent=2),
            timeout=self.request_timeout_seconds,
        )

        return extract_json_object(response)

    def _repair_missing_or_extra_keys(
        self,
        *,
        expected_ids: list[str],
        result: dict,
        batch: list[dict],
        story_idea: str,
        global_context: dict,
        notes: str,
        previous_concepts: dict,
        previous_summary: str,
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict:
        # Drop unexpected keys.
        repaired = {
            key: value
            for key, value in result.items()
            if key in expected_ids
        }

        missing = [segment_id for segment_id in expected_ids if segment_id not in repaired]

        if not missing:
            return {
                segment_id: repaired[segment_id]
                for segment_id in expected_ids
            }

        self._report(
            f"Concept batch: repairing {len(missing)} missing scene "
            f"{'key' if len(missing) == 1 else 'keys'}",
            progress_callback,
        )

        # One focused repair call for missing keys only.
        missing_segments = [
            seg
            for seg in batch
            if seg["segment_id"] in missing
        ]

        system_prompt = """
You repair missing music-video visual concepts.

Return ONLY valid JSON object.
Create exactly one concise concept for each segment in MISSING_SEGMENTS.
Each key must exactly match the segment_id.
Do not add extra keys.

Rules:
- One sentence per concept.
- Preserve story continuity.
- If GLOBAL_CONTEXT.location_constraint is provided, follow it as a mandatory rule.
- Do not describe subject identity/outfit/hair.
- Do not repeat full prompts.
- Describe visual story action, environment, transformation, symbol, or mood.
""".strip()

        payload = {
            "STORY_IDEA": story_idea,
            "GLOBAL_CONTEXT": global_context,
            "NOTES": notes,
            "PREVIOUS_PROGRESS_SUMMARY": previous_summary,
            "PREVIOUS_CONCEPTS": previous_concepts,
            "MISSING_SEGMENTS": missing_segments,
            "EXPECTED_KEYS": missing,
        }

        response = self.llm.complete_prompt(
            system_prompt=system_prompt,
            prompt=json.dumps(payload, ensure_ascii=False, indent=2),
            timeout=self.request_timeout_seconds,
        )

        repair = extract_json_object(response)

        for segment_id in missing:
            value = repair.get(segment_id)
            if value is None:
                value = self._fallback_concept(segment_id, global_context)
            repaired[segment_id] = str(value)

        return {
            segment_id: repaired[segment_id]
            for segment_id in expected_ids
        }

    def _last_concepts(self, concepts: dict[str, str]) -> dict[str, str]:
        if self.max_previous_concepts <= 0:
            return {}

        items = list(concepts.items())[-self.max_previous_concepts:]
        return dict(items)

    def _summarize_progress(
        self,
        *,
        story_idea: str,
        global_context: dict,
        concepts: dict[str, str],
    ) -> str:
        if not concepts:
            return ""

        recent = dict(list(concepts.items())[-12:])

        system_prompt = """
Summarize the current visual story progression for a music video.

Return only 2-4 concise sentences.
Focus on:
- where the story currently is
- what changed visually
- what tension or transformation is ongoing
- what should remain continuous next

Do not mention JSON or segment ids unless needed.
""".strip()

        payload = {
            "STORY_IDEA": story_idea,
            "GLOBAL_CONTEXT": global_context,
            "RECENT_CONCEPTS": recent,
        }

        return self.llm.complete_prompt(
            system_prompt=system_prompt,
            prompt=json.dumps(payload, ensure_ascii=False, indent=2),
            timeout=self.request_timeout_seconds,
        ).strip()

    @staticmethod
    def _fallback_concept(segment_id: str, global_context: dict | None = None) -> str:
        locations = (global_context or {}).get("locations") or []
        first_location = str(locations[0]).strip() if locations else ""
        if first_location:
            return (
                f"Continue the established visual story for {segment_id} in {first_location}, "
                f"preserving atmosphere, symbolic tension, and narrative progression."
            )
        return (
            f"Continue the established visual story for {segment_id}, preserving the same setting, "
            f"atmosphere, symbolic tension, and narrative progression."
        )


def save_concepts(path: str | Path, concepts: dict, *, artifact_store: ArtifactStore) -> Path:
    return artifact_store.write_json(path, concepts)
