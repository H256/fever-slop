from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from feverslop.prompting.general_signatures import (
    LyricCorrections,
    PromptResult,
    SongBriefResult,
)
from feverslop.prompting.music_video_signatures import MusicVideoSubjectLocations


@dataclass
class PromptCall:
    guide: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


class GeneralModulesFake:
    def __init__(self, *, song_brief: dict[str, Any] | None = None, lyric_alignment=None,
                 zimage: str = "T2I RESULT", i2v: str = "I2V RESULT", storyboard: str = ""):
        self.calls: list[PromptCall] = []
        self.song_brief_result = SongBriefResult.model_validate(song_brief) if song_brief else None
        self.lyric_result = LyricCorrections.model_validate(lyric_alignment or {})
        self.zimage_result = zimage
        self.i2v_result = i2v
        self.storyboard_result = storyboard

    def song_brief(self, payload: dict[str, Any]) -> SongBriefResult:
        self.calls.append(PromptCall(payload=payload))
        return self.song_brief_result

    def lyric_alignment(self, payload: dict[str, Any]) -> LyricCorrections:
        self.calls.append(PromptCall(payload=payload))
        return self.lyric_result

    def zimage_prompt(self, payload: dict[str, Any]) -> PromptResult:
        self.calls.append(PromptCall(payload=payload))
        return PromptResult(prompt=self.zimage_result)

    def i2v_prompt(self, payload: dict[str, Any], *, guide: str) -> PromptResult:
        self.calls.append(PromptCall(guide=guide, payload=payload))
        return PromptResult(prompt=self.i2v_result)

    def storyboard_transform(self, payload: dict[str, Any]) -> PromptResult:
        self.calls.append(PromptCall(payload=payload))
        return PromptResult(prompt=self.storyboard_result)


class MusicVideoModulesFake:
    def __init__(self, *, subject_locations: dict[str, Any] | None = None,
                 concepts: dict[str, Any] | None = None,
                 detail: str = "DETAIL RESULT", t2i: str = "T2I RESULT",
                 i2v: str = "I2V RESULT"):
        self.calls: list[PromptCall] = []
        self.subject_result = MusicVideoSubjectLocations.model_validate(
            subject_locations or {"subject": "a singer", "actors": [], "locations": []},
        )
        self.concepts_result = concepts or {}
        self.detail_result = detail
        self.t2i_result = t2i
        self.i2v_result = i2v

    def subject_locations(self, story_idea: str, notes: str = "") -> MusicVideoSubjectLocations:
        self.calls.append(PromptCall(payload={"story_idea": story_idea, "notes": notes}))
        return self.subject_result

    def concepts(self, payload: dict[str, Any], **_: Any) -> dict[str, Any]:
        self.calls.append(PromptCall(payload=payload))
        return self.concepts_result

    def detail(self, label: str, payload: dict[str, Any], guide: str, **_: Any) -> str:
        self.calls.append(PromptCall(guide=guide, payload={"label": label, **payload}))
        return self.detail_result

    def t2i(self, payload: dict[str, Any], guide: str) -> str:
        self.calls.append(PromptCall(guide=guide, payload=payload))
        return self.t2i_result

    def i2v(self, payload: dict[str, Any], guide: str, performance_policy: str) -> str:
        self.calls.append(PromptCall(
            guide=guide,
            payload={**payload, "performance_policy": performance_policy},
        ))
        return self.i2v_result
