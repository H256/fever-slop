from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LLMTaskPolicy:
    profile: str
    max_tokens: int
    max_words: int | None = None


_STRUCTURED = LLMTaskPolicy("structured", max_tokens=512, max_words=150)
_CREATIVE = LLMTaskPolicy("creative", max_tokens=2048, max_words=300)

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
