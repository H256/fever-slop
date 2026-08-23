from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LLMTaskPolicy:
    profile: str
    max_tokens: int


# Music-video task names mirror the music-video signature bundle keys. The
# constants live here because music_video_modules already imports this module.
STORY_IDEA = "story_idea"
STYLE_BLOCK = "style_block"
SUBJECT_LOCATIONS = "subject_locations"
CONCEPT_MAP = "concept_map"
REPAIR_CONCEPTS = "repair_concepts"
SUMMARY = "summary"
DETAIL = "detail"
T2I = "t2i"
I2V = "i2v"

_STRUCTURED = LLMTaskPolicy("structured", max_tokens=2048)
_CREATIVE = LLMTaskPolicy("creative", max_tokens=2048)

# Concept batches return one structured value per scene. The per-scene budget
# must be multiplied by the batch size because max_tokens limits the complete
# response, not each item in the response. The overhead covers JSON keys and
# delimiters; callers should not use the global llm.max_tokens for this.
CONCEPT_PER_SCENE_TOKENS = 2048
CONCEPT_BATCH_JSON_OVERHEAD = 1024
LYRIC_ALIGNMENT_PER_SEGMENT_TOKENS = 256
MSR_PER_RELAY_TOKENS = 2048

# Batched tasks are budgeted by the per-call-site multiplier functions
# (concept_batch_max_tokens / lyric_alignment_max_tokens), not by this
# static map; their entries here are only a conservative fallback.
BATCHED_TASK_NAMES = frozenset({CONCEPT_MAP, "lyric_alignment"})

_POLICIES = {
    "song_brief": _CREATIVE,
    "lyric_alignment": _STRUCTURED,
    "zimage_prompt": _STRUCTURED,
    "i2v_prompt": _STRUCTURED,
    "storyboard_transform": _STRUCTURED,
    STORY_IDEA: _CREATIVE,
    STYLE_BLOCK: _CREATIVE,
    SUBJECT_LOCATIONS: _STRUCTURED,
    CONCEPT_MAP: _STRUCTURED,
    REPAIR_CONCEPTS: _STRUCTURED,
    SUMMARY: _STRUCTURED,
    DETAIL: _STRUCTURED,
    T2I: _STRUCTURED,
    I2V: _STRUCTURED,
}


def policy_for(task: str) -> LLMTaskPolicy:
    """Return the explicit policy for a known task, or a conservative default."""
    return _POLICIES.get(str(task).strip().lower(), _STRUCTURED)


def known_task_names() -> frozenset[str]:
    """Return every task name that has an explicit policy entry."""
    return frozenset(_POLICIES)


def concept_batch_max_tokens(batch_size: int) -> int:
    """Return the complete output budget for a concept batch."""
    if batch_size < 1:
        return CONCEPT_PER_SCENE_TOKENS
    return (CONCEPT_PER_SCENE_TOKENS * batch_size) + CONCEPT_BATCH_JSON_OVERHEAD


def lyric_alignment_max_tokens(segment_count: int) -> int:
    """Return the complete output budget for lyric corrections."""
    if segment_count < 1:
        return LYRIC_ALIGNMENT_PER_SEGMENT_TOKENS
    return (LYRIC_ALIGNMENT_PER_SEGMENT_TOKENS * segment_count) + CONCEPT_BATCH_JSON_OVERHEAD


def msr_segments_max_tokens(relay_count: int) -> int:
    """Return the complete output budget for MSR relay prompts."""
    if relay_count < 1:
        return MSR_PER_RELAY_TOKENS
    return (MSR_PER_RELAY_TOKENS * relay_count) + CONCEPT_BATCH_JSON_OVERHEAD
