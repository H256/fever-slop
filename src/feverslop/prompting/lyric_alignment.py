from __future__ import annotations

import re

from feverslop.domain.timeline import TimelineSegment
from feverslop.ports.llm import LLMPort
from feverslop.prompting.general_modules import GeneralPromptModules


class LyricTimelineAligner:
    def __init__(self, llm: LLMPort, *, modules=None):
        self.llm = llm
        self._modules = modules if modules is not None else GeneralPromptModules(llm)

    def align(self, timeline: list[TimelineSegment], reference_lyrics: str) -> list[TimelineSegment]:
        reference_lyrics = _without_section_markers(reference_lyrics)
        vocal_segments = [segment for segment in timeline if segment.kind == "vocals"]
        if not reference_lyrics or not vocal_segments:
            return timeline

        payload = {
            "REFERENCE_LYRICS": reference_lyrics,
            "WHISPER_SEGMENTS": [
                {
                    "key": f"segment{index}",
                    "start": segment.start,
                    "end": segment.end,
                    "duration": round(segment.end - segment.start, 3),
                    "text": segment.text,
                }
                for index, segment in enumerate(vocal_segments, start=1)
            ],
        }
        corrected = self._modules.lyric_alignment(payload).segments
        expected_keys = [f"segment{index}" for index in range(1, len(vocal_segments) + 1)]
        actual_keys = list(corrected.keys())
        if set(actual_keys) != set(expected_keys):
            raise ValueError(
                f"Expected {len(expected_keys)} corrected lyric segments with keys "
                f"{expected_keys}, got {actual_keys}",
            )

        corrected_segments = {}
        for segment, key in zip(vocal_segments, expected_keys, strict=True):
            aligned_text = _without_section_markers(corrected[key])
            corrected_segments[id(segment)] = TimelineSegment(
                start=segment.start,
                end=segment.end,
                kind=segment.kind,
                text=aligned_text,
                word_timestamps=self._complete_word_timestamps(
                    aligned_text,
                    segment,
                ),
            )

        return [
            corrected_segments.get(id(seg), seg) for seg in timeline
        ]

    @staticmethod
    def _complete_word_timestamps(
        text: str,
        segment: TimelineSegment,
    ) -> tuple[dict[str, object], ...]:
        words = text.split()
        if not words:
            return ()

        def normalize(value: object) -> str:
            return re.sub(r"[^\w]+", "", str(value).casefold())

        timestamp_items = [
            item
            for item in segment.word_timestamps
            if normalize(item.get("word", ""))
        ]
        anchors: dict[int, tuple[float, float]] = {}
        cursor = 0
        for item in timestamp_items:
            token = normalize(item.get("word", ""))
            while cursor < len(words) and normalize(words[cursor]) != token:
                cursor += 1
            if cursor == len(words):
                break
            try:
                anchors[cursor] = (float(item["start"]), float(item["end"]))
            except (KeyError, TypeError, ValueError):
                pass
            cursor += 1

        boundaries: list[tuple[float, float] | None] = [None] * len(words)
        for index, value in anchors.items():
            boundaries[index] = value

        anchor_indexes = sorted(anchors)
        if anchor_indexes:
            first_anchor = anchor_indexes[0]
            last_anchor = anchor_indexes[-1]
            invalid_prefix = (
                first_anchor > 0
                and anchors[first_anchor][0] <= segment.start
            )
            invalid_suffix = (
                last_anchor < len(words) - 1
                and anchors[last_anchor][1] >= segment.end
            )
            invalid_gap = any(
                right > left + 1
                and anchors[right][0] <= anchors[left][1]
                for left, right in zip(anchor_indexes, anchor_indexes[1:], strict=False)
            )
            if invalid_prefix or invalid_suffix or invalid_gap:
                anchor_indexes = []
                anchors = {}
                boundaries = [None] * len(words)

        runs = []
        if anchor_indexes:
            runs.append((0, anchor_indexes[0], segment.start, anchors[anchor_indexes[0]][0]))
            for left, right in zip(anchor_indexes, anchor_indexes[1:], strict=False):
                runs.append((left + 1, right, anchors[left][1], anchors[right][0]))
            runs.append((anchor_indexes[-1] + 1, len(words), anchors[anchor_indexes[-1]][1], segment.end))
        else:
            runs.append((0, len(words), segment.start, segment.end))

        for start, end, range_start, range_end in runs:
            count = end - start
            if count <= 0:
                continue
            duration = max(0.0, range_end - range_start)
            step = duration / count if count else 0.0
            for offset in range(count):
                index = start + offset
                if boundaries[index] is None:
                    word_start = range_start + step * offset
                    boundaries[index] = (word_start, word_start + step)

        return tuple(
            {
                "word": word,
                "start": round(boundary[0], 3),
                "end": round(boundary[1], 3),
            }
            for word, boundary in zip(words, boundaries, strict=True)
            if boundary is not None
        )


def _without_section_markers(value: object) -> str:
    return re.sub(r"(?m)^\s*\[[^]\r\n]+\]\s*$", "", str(value or "")).strip()
