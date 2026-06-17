from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any

from llm_client import LocalOpenAIClient


def extract_json_object(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in LLM response:\n{text}")

    return json.loads(text[start:end + 1])


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
        llm: LocalOpenAIClient,
        batch_size: int = 10,
        max_previous_concepts: int = 6,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        self.llm = llm
        self.batch_size = batch_size
        self.max_previous_concepts = max_previous_concepts

    def create_concept_prompts_batched(
        self,
        *,
        stage1_segments: list[dict],
        story_idea: str,
        global_context: dict,
        notes: str = "",
    ) -> dict:
        all_results: dict[str, str] = {}
        previous_summary = ""

        for batch_start, batch in chunked(stage1_segments, self.batch_size):
            batch_result = self._generate_batch(
                batch_index=batch_start // self.batch_size + 1,
                batch=batch,
                story_idea=story_idea,
                global_context=global_context,
                notes=notes,
                previous_concepts=self._last_concepts(all_results),
                previous_summary=previous_summary,
            )

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
            )

            all_results.update(batch_result)

            previous_summary = self._summarize_progress(
                story_idea=story_idea,
                global_context=global_context,
                concepts=all_results,
            )

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
        system_prompt = """
You are a music-video visual concept mapper.

You receive only one batch of timed song sections, but the whole video must remain continuous.

TASK:
Create exactly one concise visual concept for each segment in CURRENT_BATCH_SEGMENTS.

Rules:
- Return ONLY valid JSON object.
- Each key must exactly match a segment_id from CURRENT_BATCH_SEGMENTS.
- Do not omit any segment_id.
- Do not add extra keys.
- Concepts are story beats, not final prompts.
- Do NOT describe subject details, outfit, hair, age, ethnicity, or identity. The subject is injected later.
- Do include visible setting, action, props, transformation, atmosphere, symbolic events, environmental motion, and emotional story progression.
- For instrumental segments: advance the visual story without singing.
- For vocals/mixed segments: reflect lyrics while preserving continuity.
- Keep each concept one sentence.
- Avoid "lip-sync" and "sings" in concepts unless the segment's state strongly requires vocal performance.
- Maintain continuity with PREVIOUS_CONCEPTS and PREVIOUS_PROGRESS_SUMMARY.
- No markdown, no comments.
""".strip()

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
            system_prompt=system_prompt,
            prompt=json.dumps(payload, ensure_ascii=False, indent=2),
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
    ) -> dict:
        # Drop unexpected keys.
        repaired = {
            key: str(value)
            for key, value in result.items()
            if key in expected_ids
        }

        missing = [segment_id for segment_id in expected_ids if segment_id not in repaired]

        if not missing:
            return {
                segment_id: repaired[segment_id]
                for segment_id in expected_ids
            }

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
        )

        repair = extract_json_object(response)

        for segment_id in missing:
            value = repair.get(segment_id)
            if value is None:
                value = self._fallback_concept(segment_id)
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
        ).strip()

    @staticmethod
    def _fallback_concept(segment_id: str) -> str:
        return (
            f"Continue the established visual story for {segment_id}, preserving the same setting, "
            f"atmosphere, symbolic tension, and narrative progression."
        )


def save_concepts(path: str | Path, concepts: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(concepts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
