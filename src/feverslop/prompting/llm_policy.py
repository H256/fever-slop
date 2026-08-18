from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LLMTaskPolicy:
    profile: str
    max_tokens: int
    max_words: int | None = None


_STRUCTURED = LLMTaskPolicy("structured", max_tokens=512, max_words=150)
_CREATIVE = LLMTaskPolicy("creative", max_tokens=2048, max_words=300)

# Concept batches return one structured value per scene. The per-scene budget
# must be multiplied by the batch size because max_tokens limits the complete
# response, not each item in the response. The overhead covers JSON keys and
# delimiters; callers should not use the global llm.max_tokens for this.
CONCEPT_PER_SCENE_TOKENS = 512
CONCEPT_BATCH_JSON_OVERHEAD = 1024
LYRIC_ALIGNMENT_PER_SEGMENT_TOKENS = 256
MSR_PER_RELAY_TOKENS = 512

_POLICIES = {
    "song_brief": _CREATIVE,
    "movie_story_arch": _CREATIVE,
    "movie_bible": _CREATIVE,
    "lyric_alignment": _STRUCTURED,
    "zimage_prompt": _STRUCTURED,
    "i2v_prompt": _STRUCTURED,
    "storyboard_transform": _STRUCTURED,
}


def policy_for(task: str) -> LLMTaskPolicy:
    """Return the conservative policy for an unknown signature."""
    return _POLICIES.get(str(task).strip().lower(), _STRUCTURED)


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
