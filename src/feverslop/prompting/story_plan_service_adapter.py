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
) -> str:
    parts: list[str] = [
        f"Title: {song_title}",
        f"Language: {song_language}",
    ]
    if song_style:
        parts.append(f"Style: {song_style}")
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
        result = self._modules.bible(
            story_text=_build_story_text(
                song_title=song_title,
                song_language=song_language,
                song_style=song_style,
                lyrics=lyrics,
                sections=list(sections),
            ),
            creative_direction=self._creative_direction,
            characters=[dict(character) for character in characters],
            locations=[],
            props=[],
        )
        self._last_bible = result if isinstance(result, Mapping) else {}
        return result

    def beat_allocation(
        self,
        *,
        song_title: str,
        lyrics: str,
        segments: list[Mapping[str, Any]],
        narrative_bible: Mapping[str, Any],
        characters: list[Mapping[str, Any]],
        terminal_window_seconds: float,
        guide: str,
        **_extra: Any,
    ) -> Any:
        return self._modules.beat_allocation(
            creative_direction=self._creative_direction,
            bible=dict(narrative_bible),
            characters=[dict(c) for c in characters],
            locations=[],
            props=[],
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
        **_extra: Any,
    ) -> Any:
        return self._modules.acting(
            creative_direction=self._creative_direction,
            bible=dict(self._last_bible),
            beats=[dict(b) for b in briefs],
            segments=[],
            characters=[dict(c) for c in characters],
            locations=[],
            props=[],
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
