from __future__ import annotations

import re

from feverslop.domain.timeline import TimelineSegment
from feverslop.domain.word_alignment import align_words, alignment_fingerprint, raw_word_rows
from feverslop.utils.sub_step_progress import SubStepProgress
from feverslop.ports.llm import LLMPort
from feverslop.prompting.general_modules import GeneralPromptModules


class LyricTimelineAligner:
    def __init__(self, llm: LLMPort, *, modules=None):
        self.llm = llm
        self._modules = modules if modules is not None else GeneralPromptModules(llm)
        self.reporter = None

    def set_reporter(self, reporter):
        self.reporter = reporter

    def align(self, timeline: list[TimelineSegment], reference_lyrics: str) -> list[TimelineSegment]:
        reference_lyrics = _without_section_markers(reference_lyrics)
        vocal_segments = [segment for segment in timeline if segment.kind == "vocals"]
        if not reference_lyrics or not vocal_segments:
            return timeline

        raw = [{"start": float(seg.start), "end": float(seg.end),
                "text": (seg.alignment or {}).get("raw_text", seg.text),
                "words": (seg.alignment or {}).get("raw_words", raw_word_rows(
                    seg.word_timestamps, start=seg.start, end=seg.end))}
               for seg in vocal_segments]
        fingerprint = alignment_fingerprint(raw, reference_lyrics)
        progress = SubStepProgress(self.reporter, "Word timing alignment", len(vocal_segments))
        progress.update(0, force=True, detail=f"source_words={sum(len(seg['words']) for seg in raw)}")
        if all((seg.alignment or {}).get("fingerprint") == fingerprint for seg in vocal_segments):
            progress.update(len(vocal_segments), detail="reused unchanged alignment")
            return timeline

        payload = {
            "REFERENCE_LYRICS": reference_lyrics,
            "WHISPER_SEGMENTS": [
                {
                    "key": f"segment{index}",
                    "start": segment.start,
                    "end": segment.end,
                    "duration": round(segment.end - segment.start, 3),
                    "text": raw[index - 1]["text"],
                }
                for index, segment in enumerate(vocal_segments, start=1)
            ],
        }
        corrected = self._modules.lyric_alignment(payload).segments
        expected_keys = [f"segment{index}" for index in range(1, len(vocal_segments) + 1)]
        actual_keys = list(corrected.keys())
        if not corrected:
            # A malformed/empty structured response should not discard valid
            # Whisper timing evidence or abort an otherwise resumable run.
            corrected = {
                key: raw[index - 1]["text"]
                for index, key in enumerate(expected_keys, start=1)
            }
            actual_keys = expected_keys
        if set(actual_keys) != set(expected_keys):
            raise ValueError(
                f"Expected {len(expected_keys)} corrected lyric segments with keys "
                f"{expected_keys}, got {actual_keys}",
            )

        corrected_segments = {}
        timed_count = unresolved_count = invalid_count = 0
        for index, (segment, key) in enumerate(zip(vocal_segments, expected_keys, strict=True)):
            aligned_text = _without_section_markers(corrected[key])
            alignment = align_words(raw[index]["text"], raw[index]["words"], aligned_text,
                                    segment.start, segment.end)
            alignment["fingerprint"] = fingerprint
            corrected_segments[id(segment)] = TimelineSegment(
                start=segment.start,
                end=segment.end,
                kind=segment.kind,
                text=aligned_text,
                evidence=segment.evidence,
                word_timestamps=tuple(alignment["timed_words"]),
                alignment=alignment,
            )
            timed_count += len(alignment["timed_words"])
            unresolved_count += len(alignment["targets"]) - len(alignment["timed_words"])
            invalid_count += len(alignment["diagnostics"])
            progress.update(index + 1, detail=(f"timed={timed_count}, "
                            f"unresolved={unresolved_count}, invalid={invalid_count}"))

        return [
            corrected_segments.get(id(seg), seg) for seg in timeline
        ]

def _without_section_markers(value: object) -> str:
    return re.sub(r"(?m)^\s*\[[^]\r\n]+\]\s*$", "", str(value or "")).strip()
