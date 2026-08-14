from __future__ import annotations

from feverslop.domain.llm_parsing import extract_json_object
from feverslop.domain.timeline import TimelineSegment
from feverslop.ports.llm import LLMPort
from feverslop.prompting.general_modules import GeneralPromptModules


LYRIC_ALIGNMENT_SYSTEM_PROMPT = """
You correct Whisper lyric transcription segments using complete reference lyrics.

Rules:
- Return valid JSON only.
- Return exactly one output value for each input segment.
- Keep segment numbering unchanged.
- Do not merge, split, skip, reorder, or move words between segments.
- Preserve the rough amount of text that fits the segment duration.
- Fix misheard words, punctuation, capitalization, and missing short words using the reference lyrics.
- If the Whisper segment is empty or obviously wrong, infer the most likely lyric phrase for that same position in the song.
- Do not add explanations, markdown, comments, or extra fields.

Output shape:
{
  "segment1": "corrected lyrics for input segment 1",
  "segment2": "corrected lyrics for input segment 2"
}
""".strip()


class LyricTimelineAligner:
    def __init__(self, llm: LLMPort):
        self.llm = llm
        self._modules = GeneralPromptModules(llm)

    def align(self, timeline: list[TimelineSegment], reference_lyrics: str) -> list[TimelineSegment]:
        reference_lyrics = str(reference_lyrics or "").strip()
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
        result = self._modules.lyric_alignment(payload, legacy_system_prompt=LYRIC_ALIGNMENT_SYSTEM_PROMPT)
        corrected = result.segments if hasattr(result, "segments") else extract_json_object(result)
        expected_keys = [f"segment{index}" for index in range(1, len(vocal_segments) + 1)]
        actual_keys = list(corrected.keys())
        if actual_keys != expected_keys:
            raise ValueError(
                f"Expected {len(expected_keys)} corrected lyric segments with keys "
                f"{expected_keys}, got {actual_keys}"
            )

        corrected_segments = {}
        for segment, key in zip(vocal_segments, expected_keys, strict=True):
            corrected_segments[id(segment)] = TimelineSegment(
                start=segment.start,
                end=segment.end,
                kind=segment.kind,
                text=str(corrected[key]).strip(),
                word_timestamps=segment.word_timestamps,
            )

        return [
            corrected_segments.get(id(seg), seg) for seg in timeline
        ]
