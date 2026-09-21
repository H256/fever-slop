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
NARRATIVE_CONTRACT = "narrative_contract"
REPAIR_CONCEPTS = "repair_concepts"
SONG_BRIEF = "song_brief"
STORYBOARD_TRANSFORM = "storyboard_transform"
STORY_IDEA = "story_idea"
STORY_PLAN_ACTING = "story_plan_acting"
STORY_PLAN_BEAT_ALLOCATION = "story_plan_beat_allocation"
STORY_PLAN_BIBLE = "story_plan_bible"
STORY_PLAN_REPAIR = "story_plan_repair"
STYLE_BLOCK = "style_block"
SUBJECT_LOCATIONS = "subject_locations"
SUMMARY = "summary"
T2I = "t2i"
ZIMAGE_PROMPT = "zimage_prompt"

# set some limits for structured and creative tasks
_STRUCTURED = LLMTaskPolicy("structured", max_tokens=2048)
_CREATIVE = LLMTaskPolicy("creative", max_tokens=2048)

# Story-plan jobs return typed planning output. The bible carries notes for
# many entities and the beat allocation carries per-brief constraints for
# ~10-20 segments; 2048 truncates verbose small models (same reasoning as
# the H3 planner comment). The repair must emit the complete plan, so it
# mirrors H3_PLANNER_MAX_TOKENS.
_STORY_PLAN_STRUCTURED = LLMTaskPolicy("structured", max_tokens=4096)
_STORY_PLAN_CREATIVE = LLMTaskPolicy("creative", max_tokens=4096)
_STORY_PLAN_REPAIR = LLMTaskPolicy("structured", max_tokens=8192)

# Concept batches return one structured value per scene. The per-scene budget
# must be multiplied by the batch size because max_tokens limits the complete
# response, not each item in the response. The overhead covers JSON keys and
# delimiters; callers should not use the global llm.max_tokens for this.
CONCEPT_PER_SCENE_TOKENS = 2048
CONCEPT_BATCH_JSON_OVERHEAD = 2048
LYRIC_ALIGNMENT_PER_SEGMENT_TOKENS = 1024
MSR_PER_RELAY_TOKENS = 2048

# H3 is called once per scene. Never inherit the application's large global
# response budget for these bounded structured/advisory tasks.
#
# The planner's typed plan (creative_intent, style_opening, overall_soundscape
# and one planned shot per authoritative relay segment, each carrying several
# prose fields) is closer to 6-8k tokens than 4k when the model is verbose. At
# 4096 the response is frequently truncated mid-JSON on weaker/quantized models,
# which surfaces downstream as h3.fallback.plan_missing. The budget is component
# configurable via `prompt_planner_max_tokens` on the LLM override.
H3_PLANNER_MAX_TOKENS = 8192
H3_JUDGE_MAX_TOKENS = 2048

# Auto-scaling constants used when `prompt_planner_max_tokens` is 0 (default).
# PLANNER_TOKEN_OVERHEAD covers JSON structural tokens that don't scale with scenes.
# PLANNER_TOKEN_PER_SHOT covers prose fields per authoritative relay segment (shot).
PLANNER_TOKEN_OVERHEAD = 4096
PLANNER_TOKEN_PER_SHOT = 1536

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
    NARRATIVE_CONTRACT: _STRUCTURED,
    REPAIR_CONCEPTS: _STRUCTURED,
    SONG_BRIEF: _CREATIVE,
    STORYBOARD_TRANSFORM: _STRUCTURED,
    STORY_IDEA: _CREATIVE,
    STORY_PLAN_ACTING: _STORY_PLAN_CREATIVE,
    STORY_PLAN_BEAT_ALLOCATION: _STORY_PLAN_STRUCTURED,
    STORY_PLAN_BIBLE: _STORY_PLAN_STRUCTURED,
    STORY_PLAN_REPAIR: _STORY_PLAN_REPAIR,
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
