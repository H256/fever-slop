from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError


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
    vocal_performers: list["VocalPerformer"] = Field(default_factory=list)


class VocalPerformer(BaseModel):
    """A visible vocal source chosen by the creative prompt model."""

    subject_id: str = Field(min_length=1)
    speaker_id: str = Field(pattern=r"^S[1-9][0-9]*$")


class StoryboardPromptResult(PromptResult):
    """Transport model; the transformer applies the configured word limit."""


def parse_prompt_result(value: Any) -> PromptResult:
    """Validate an LLM prompt result without treating optional performer hints as fatal."""
    try:
        return PromptResult.model_validate(value)
    except ValidationError as error:
        if not isinstance(value, Mapping) or not _only_vocal_performer_errors(error):
            raise
        sanitized = dict(value)
        sanitized["vocal_performers"] = _valid_vocal_performers(value.get("vocal_performers"))
        return PromptResult.model_validate(sanitized)


def _only_vocal_performer_errors(error: ValidationError) -> bool:
    return all(
        issue.get("loc", ())[0:1] == ("vocal_performers",)
        for issue in error.errors()
    )


def _valid_vocal_performers(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    performers = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        subject_id = str(item.get("subject_id") or "").strip()
        match = re.fullmatch(r"[sS]([1-9][0-9]*)", str(item.get("speaker_id") or "").strip())
        if subject_id and match:
            performers.append({"subject_id": subject_id, "speaker_id": f"S{match.group(1)}"})
    return performers


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
        """Write one image-to-video prompt and identify any visible vocal performers by subject ID."""

        guide: str = dspy_module.InputField()
        payload: dict[str, Any] = dspy_module.InputField()
        result: dict[str, Any] = dspy_module.OutputField()

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
