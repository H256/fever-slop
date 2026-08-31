from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LLMTaskPolicy:
    profile: str
    max_tokens: int


# Music-video task names mirror the music-video signature bundle keys. The
# constants live here because music_video_modules already imports this module.
CONCEPT_MAP = "concept_map"
DETAIL = "detail"
I2V = "i2v"
I2V_PROMPT = "i2v_prompt"
LYRIC_ALIGNMENT = "lyric_alignment"
REPAIR_CONCEPTS = "repair_concepts"
SONG_BRIEF = "song_brief"
STORYBOARD_TRANSFORM = "storyboard_transform"
STORY_IDEA = "story_idea"
STYLE_BLOCK = "style_block"
SUBJECT_LOCATIONS = "subject_locations"
SUMMARY = "summary"
T2I = "t2i"
ZIMAGE_PROMPT = "zimage_prompt"

# set some limits for structured and creative tasks
_STRUCTURED = LLMTaskPolicy("structured", max_tokens=2048)
_CREATIVE = LLMTaskPolicy("creative", max_tokens=2048)

# Concept batches return one structured value per scene. The per-scene budget
# must be multiplied by the batch size because max_tokens limits the complete
# response, not each item in the response. The overhead covers JSON keys and
# delimiters; callers should not use the global llm.max_tokens for this.
CONCEPT_PER_SCENE_TOKENS = 2048
CONCEPT_BATCH_JSON_OVERHEAD = 2048
LYRIC_ALIGNMENT_PER_SEGMENT_TOKENS = 1024
MSR_PER_RELAY_TOKENS = 2048

# Batched tasks are budgeted by the per-call-site multiplier functions
# (concept_batch_max_tokens / lyric_alignment_max_tokens), not by this
# static map; their entries here are only a conservative fallback.
BATCHED_TASK_NAMES = frozenset({CONCEPT_MAP, "lyric_alignment"})

_POLICIES = {
    CONCEPT_MAP: _STRUCTURED,
    DETAIL: _STRUCTURED,
    I2V: _STRUCTURED,
    I2V_PROMPT: _STRUCTURED,
    LYRIC_ALIGNMENT: _STRUCTURED,
    REPAIR_CONCEPTS: _STRUCTURED,
    SONG_BRIEF: _CREATIVE,
    STORYBOARD_TRANSFORM: _STRUCTURED,
    STORY_IDEA: _CREATIVE,
    STYLE_BLOCK: _CREATIVE,
    SUBJECT_LOCATIONS: _STRUCTURED,
    SUMMARY: _STRUCTURED,
    T2I: _STRUCTURED,
    ZIMAGE_PROMPT: _STRUCTURED,
}


def _calculate_batch_token_budget(count: int, tokens_per_item: int,
                                  overhead_tokens: int = CONCEPT_BATCH_JSON_OVERHEAD) -> int:
    """Calculate total token budget ensuring at least a single-item budget plus overhead."""
    effective_count = max(1, count)
    return (tokens_per_item * effective_count) + overhead_tokens


def policy_for(task: str) -> LLMTaskPolicy:
    """Return the explicit policy for a known task, or a conservative default."""
    return _POLICIES.get(str(task).strip().lower(), _STRUCTURED)


def concept_batch_max_tokens(batch_size: int) -> int:
    """Return the complete output budget for a concept batch."""
    return _calculate_batch_token_budget(batch_size, CONCEPT_PER_SCENE_TOKENS)


def lyric_alignment_max_tokens(segment_count: int) -> int:
    """Return the complete output budget for lyric corrections."""
    return _calculate_batch_token_budget(segment_count, LYRIC_ALIGNMENT_PER_SEGMENT_TOKENS)


def msr_segments_max_tokens(relay_count: int) -> int:
    """Return the complete output budget for MSR relay prompts."""
    return _calculate_batch_token_budget(relay_count, MSR_PER_RELAY_TOKENS)
