from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SongBriefResult(BaseModel):
    title: str
    tags: str
    lyrics: str
    bpm: int
    language: str
    keyscale: str
    visual_story_idea: str
    visual_style: str


class LyricCorrections(BaseModel):
    segments: dict[str, str] = Field(default_factory=dict)


class PromptResult(BaseModel):
    prompt: str


class StoryboardPromptResult(PromptResult):
    """Transport model; the transformer applies the configured word limit."""


def build_general_signature_bundle(dspy_module: Any | None = None):
    if dspy_module is None:
        import dspy as dspy_module

    class SongBrief(dspy_module.Signature):
        """Create the structured ACE-Step and visual song brief."""

        guide: str = dspy_module.InputField()
        request: dict[str, Any] = dspy_module.InputField()
        result: SongBriefResult = dspy_module.OutputField()

    class LyricAlignment(dspy_module.Signature):
        """Correct transcription without changing segment boundaries or order."""

        guide: str = dspy_module.InputField()
        request: dict[str, Any] = dspy_module.InputField()
        result: LyricCorrections = dspy_module.OutputField()

    class ZImagePrompt(dspy_module.Signature):
        """Write one still-image prompt from the supplied scene payload."""

        guide: str = dspy_module.InputField()
        payload: dict[str, Any] = dspy_module.InputField()
        result: PromptResult = dspy_module.OutputField()

    class I2VPrompt(dspy_module.Signature):
        """Write one image-to-video prompt from the supplied scene payload."""

        guide: str = dspy_module.InputField()
        payload: dict[str, Any] = dspy_module.InputField()
        result: PromptResult = dspy_module.OutputField()

    class StoryboardTransform(dspy_module.Signature):
        """Transform a storyboard prompt using the supplied editable template."""

        guide: str = dspy_module.InputField()
        system_template: str = dspy_module.InputField()
        user_template: str = dspy_module.InputField()
        width: int = dspy_module.InputField()
        height: int = dspy_module.InputField()
        original_prompt: str = dspy_module.InputField()
        max_words: int = dspy_module.InputField()
        result: StoryboardPromptResult = dspy_module.OutputField()

    return {
        "song_brief": SongBrief,
        "lyric_alignment": LyricAlignment,
        "zimage_prompt": ZImagePrompt,
        "i2v_prompt": I2VPrompt,
        "storyboard_transform": StoryboardTransform,
    }
