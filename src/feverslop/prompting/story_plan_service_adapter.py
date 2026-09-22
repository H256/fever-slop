"""Adapter that bridges the StoryPlanService job interface to StoryPlanPromptModules.

The service calls its injected prompt modules with a flat, song-oriented
payload (``song_title``, ``lyrics``, ``sections``, ``source_evidence``,
``guide``).  The typed DSPy modules expect a structured signature
(``story_text``, ``creative_direction``, ``characters``, ``locations``,
``props``).  This adapter translates between the two so the service can
drive the real modules without modification.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def _build_story_text(
    *,
    song_title: str,
    song_language: str,
    song_style: str,
    lyrics: str,
    sections: list[Mapping[str, Any]],
    story_idea: str = "",
) -> str:
    parts: list[str] = [
        f"Title: {song_title}",
        f"Language: {song_language}",
    ]
    if song_style:
        parts.append(f"Style: {song_style}")
    if story_idea:
        parts.append(f"Story idea:\n{story_idea}")
    if lyrics:
        parts.append(f"Lyrics:\n{lyrics}")
    if sections:
        parts.append(f"Sections: {json.dumps(sections, ensure_ascii=False)}")
    return "\n".join(parts)


class StoryPlanServiceAdapter:
    """Translates the service's flat job payloads to the typed module signature."""

    def __init__(self, modules: Any) -> None:
        self._modules = modules
        self._creative_direction: str = ""
        self._last_bible: dict[str, Any] = {}

    def bible(
        self,
        *,
        song_title: str,
        song_language: str,
        song_style: str,
        lyrics: str,
        sections: list[Mapping[str, Any]],
        characters: list[Mapping[str, Any]],
        source_evidence: Mapping[str, Any],
        guide: str,
        **_extra: Any,
    ) -> Any:
        self._creative_direction = str(
            source_evidence.get("creative_direction", "")
        ) if isinstance(source_evidence, Mapping) else ""
        story_idea = str(source_evidence.get("story_idea", "")) if isinstance(source_evidence, Mapping) else ""
        locations = list(source_evidence.get("locations", ())) if isinstance(source_evidence, Mapping) else []
        props = list(source_evidence.get("props", ())) if isinstance(source_evidence, Mapping) else []
        result = self._modules.bible(
            story_text=_build_story_text(
                song_title=song_title,
                song_language=song_language,
                song_style=song_style,
                lyrics=lyrics,
                sections=list(sections),
                story_idea=story_idea,
            ),
            creative_direction=self._creative_direction,
            characters=[dict(character) for character in characters],
            locations=[dict(location) for location in locations if isinstance(location, Mapping)],
            props=[dict(prop) for prop in props if isinstance(prop, Mapping)],
        )
        self._last_bible = result if isinstance(result, Mapping) else {}
        return result

    def arc_skeleton(
        self,
        *,
        song_title: str,
        lyrics: str,
        narrative_bible: Mapping[str, Any],
        characters: list[Mapping[str, Any]],
        terminal_window_seconds: float,
        guide: str,
        locations: list[Mapping[str, Any]] | None = None,
        props: list[Mapping[str, Any]] | None = None,
        **_extra: Any,
    ) -> Any:
        return self._modules.arc_skeleton(
            creative_direction=self._creative_direction,
            bible=dict(narrative_bible),
            characters=[dict(c) for c in characters],
            locations=[dict(location) for location in (locations or [])],
            props=[dict(prop) for prop in (props or [])],
        )

    def beat_allocation(
        self,
        *,
        song_title: str,
        lyrics: str,
        beats: list[Mapping[str, Any]],
        segments: list[Mapping[str, Any]],
        narrative_bible: Mapping[str, Any],
        characters: list[Mapping[str, Any]],
        terminal_window_seconds: float,
        guide: str,
        locations: list[Mapping[str, Any]] | None = None,
        props: list[Mapping[str, Any]] | None = None,
        **_extra: Any,
    ) -> Any:
        return self._modules.beat_allocation(
            creative_direction=self._creative_direction,
            bible=dict(narrative_bible),
            beats=[dict(beat) for beat in beats],
            characters=[dict(c) for c in characters],
            locations=[dict(location) for location in (locations or [])],
            props=[dict(prop) for prop in (props or [])],
            segments=[dict(s) for s in segments],
        )

    def acting(
        self,
        *,
        song_title: str,
        song_language: str,
        lyrics: str,
        briefs: list[Mapping[str, Any]],
        characters: list[Mapping[str, Any]],
        guide: str,
        segments: list[Mapping[str, Any]] | None = None,
        locations: list[Mapping[str, Any]] | None = None,
        props: list[Mapping[str, Any]] | None = None,
        typed_beats: list[Mapping[str, Any]] | None = None,
        expected_brief_ids: list[str] | None = None,
        **_extra: Any,
    ) -> Any:
        return self._modules.acting(
            creative_direction=self._creative_direction,
            bible=dict(self._last_bible),
            beats=[
                {"beat_id": f"beat-{index:03d}", **dict(beat)}
                for index, beat in enumerate(typed_beats or briefs, start=1)
            ],
            expected_brief_ids=list(expected_brief_ids or []),
            segments=[dict(segment) for segment in (segments or [])],
            characters=[dict(c) for c in characters],
            locations=[dict(location) for location in (locations or [])],
            props=[dict(prop) for prop in (props or [])],
        )

    def live_prompts(
        self,
        *,
        creative_direction: str,
        bible: Mapping[str, Any],
        briefs: list[Mapping[str, Any]],
        expected_targets: list[str],
        **_extra: Any,
    ) -> Any:
        return self._modules.live_prompts(
            creative_direction=creative_direction,
            bible=dict(bible),
            briefs=[dict(brief) for brief in briefs],
            expected_targets=list(expected_targets),
        )

    def repair(
        self,
        *,
        song_title: str,
        song_style: str,
        lyrics: str,
        candidate: Mapping[str, Any],
        validation_errors: list[Mapping[str, Any]],
        guide: str,
        **_extra: Any,
    ) -> Any:
        return self._modules.repair(
            prior_plan=dict(candidate),
            diagnostics=[dict(e) for e in validation_errors],
        )
