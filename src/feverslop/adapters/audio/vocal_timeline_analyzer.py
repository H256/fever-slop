from __future__ import annotations

import gc
from pathlib import Path

import librosa
import numpy as np
import torch
import whisper

from feverslop.domain.timeline import TimelineSegment
from feverslop.domain.timeline_transform import (
    merge_same_kind_segments,
    normalize_empty_vocals,
)

__all__ = [
    "VocalTimelineAnalyzer",
    "merge_same_kind_segments",
    "normalize_empty_vocals",
]


class VocalTimelineAnalyzer:
    def __init__(
        self,
        whisper_model: str = "large-v3",
        language: str = "de",
        silence_threshold_db: float | None = None,  # bleibt kompatibel, wird aber als RMS-Threshold ignoriert
        merge_gap: float = 0.5,
        min_vocal_duration: float = 0.4,
        min_silence_duration: float = 0.8,
        frame_length: int = 2048,
        hop_length: int = 512,
        rms_low_percentile: float = 20.0,
        rms_high_percentile: float = 85.0,
        rms_ratio: float = 0.35,
        smooth_frames: int = 10,
    ):
        self.whisper_model = whisper_model
        self.model = None
        self.raw_whisper_segments: list[dict] = []
        self.language = language
        self.merge_gap = merge_gap
        self.min_vocal_duration = min_vocal_duration
        self.min_silence_duration = min_silence_duration
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.rms_low_percentile = rms_low_percentile
        self.rms_high_percentile = rms_high_percentile
        self.rms_ratio = rms_ratio
        self.smooth_frames = smooth_frames

    def close(self) -> None:
        model = getattr(self, "model", None)
        if model is None:
            return

        self.model = None
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def analyze(self, vocals_file: str | Path) -> list[TimelineSegment]:
        vocals_file = Path(vocals_file)

        whisper_segments = self._transcribe(vocals_file)
        vocal_ranges, duration = self._detect_vocal_activity_rms(vocals_file)

        vocal_segments = self._combine_whisper_and_energy(
            whisper_segments,
            vocal_ranges,
        )

        return self._insert_instrumental_segments(
            vocal_segments,
            duration,
        )

    def _transcribe(self, vocals_file: Path) -> list[dict]:
        if self.model is None:
            self.model = whisper.load_model(self.whisper_model)
        result = self.model.transcribe(
            str(vocals_file),
            language=self.language,
            task="transcribe",
            verbose=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.75,
            logprob_threshold=-0.5,
            compression_ratio_threshold=2.0,
            temperature=0,
            word_timestamps=True,
        )

        raw_segments = list(result.get("segments") or [])
        self.raw_whisper_segments = raw_segments
        segments = []

        for s in raw_segments:
            text = s["text"].strip()
            lower = text.lower()

            if not text:
                continue

            if "untertitelung des zdf" in lower:
                continue

            if s.get("no_speech_prob", 0) > 0.85:
                continue

            segments.append(s)

        return segments

    def _detect_vocal_activity_rms(
        self,
        vocals_file: Path,
    ) -> tuple[list[tuple[float, float]], float]:
        y, sr = librosa.load(str(vocals_file), sr=None, mono=True)

        duration = float(len(y) / sr)

        rms = librosa.feature.rms(
            y=y,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[0]

        if rms.size == 0:
            return [], duration

        if self.smooth_frames > 1:
            kernel = np.ones(self.smooth_frames) / self.smooth_frames
            rms = np.convolve(rms, kernel, mode="same")

        low = float(np.percentile(rms, self.rms_low_percentile))
        high = float(np.percentile(rms, self.rms_high_percentile))

        threshold = low + ((high - low) * self.rms_ratio)

        times = librosa.frames_to_time(
            np.arange(len(rms)),
            sr=sr,
            hop_length=self.hop_length,
        )

        active = rms > threshold

        ranges = []
        active_start = None

        for i, is_active in enumerate(active):
            t = float(times[i])

            if is_active and active_start is None:
                active_start = t

            elif not is_active and active_start is not None:
                end = t
                if end - active_start >= self.min_vocal_duration:
                    ranges.append((active_start, end))
                active_start = None

        if active_start is not None:
            end = duration
            if end - active_start >= self.min_vocal_duration:
                ranges.append((active_start, end))

        ranges = self._merge_ranges(ranges, self.merge_gap)

        return ranges, duration

    @staticmethod
    def _merge_ranges(
        ranges: list[tuple[float, float]],
        max_gap: float,
    ) -> list[tuple[float, float]]:
        if not ranges:
            return []

        merged = [ranges[0]]

        for start, end in ranges[1:]:
            last_start, last_end = merged[-1]

            if start - last_end <= max_gap:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))

        return merged

    def _combine_whisper_and_energy(
        self,
        whisper_segments: list[dict],
        vocal_ranges: list[tuple[float, float]],
    ) -> list[TimelineSegment]:
        assigned_words: list[list[dict]] = [[] for _ in vocal_ranges]
        fallback_texts: list[list[str]] = [[] for _ in vocal_ranges]

        for ws in whisper_segments:
            ws_start = float(ws["start"])
            ws_end = float(ws["end"])
            words = ws.get("words") or []

            if not words:
                for index, (start, end) in enumerate(vocal_ranges):
                    if ws_start < end and ws_end > start:
                        fallback_texts[index].append(str(ws.get("text", "")).strip())
                continue

            for word in words:
                text = str(word.get("word", "")).strip()
                if not text:
                    continue

                word_start = float(word.get("start", 0))
                word_end = float(word.get("end", 0))
                overlaps = [
                    max(0.0, min(word_end, end) - max(word_start, start))
                    for start, end in vocal_ranges
                ]
                best_overlap = max(overlaps, default=0.0)
                if best_overlap <= 0:
                    continue

                best_index = min(
                    (
                        index
                        for index, overlap in enumerate(overlaps)
                        if overlap == best_overlap
                    ),
                    key=lambda index: abs(
                        (word_start + word_end) / 2
                        - (vocal_ranges[index][0] + vocal_ranges[index][1]) / 2,
                    ),
                )
                assigned_words[best_index].append(
                    {
                        "word": text,
                        "start": round(word_start, 3),
                        "end": round(word_end, 3),
                    },
                )

        result = []
        for index, (start, end) in enumerate(vocal_ranges):
            words = sorted(
                assigned_words[index], key=lambda word: (word["start"], word["end"]),
            )
            text = " ".join(word["word"] for word in words)
            if not words:
                text = " ".join(fallback_texts[index]).strip()

            result.append(
                TimelineSegment(
                    start=round(float(start), 2),
                    end=round(float(end), 2),
                    kind="vocals",
                    text=text,
                    word_timestamps=tuple(words),
                ),
            )

        return result

    def _insert_instrumental_segments(
        self,
        vocal_segments: list[TimelineSegment],
        total_duration: float,
    ) -> list[TimelineSegment]:
        timeline = []
        cursor = 0.0

        for seg in vocal_segments:
            if seg.start - cursor >= self.min_silence_duration:
                timeline.append(
                    TimelineSegment(
                        start=round(cursor, 2),
                        end=round(seg.start, 2),
                        kind="instrumental",
                    ),
                )

            timeline.append(seg)
            cursor = max(cursor, seg.end)

        if total_duration - cursor >= self.min_silence_duration:
            timeline.append(
                TimelineSegment(
                    start=round(cursor, 2),
                    end=round(total_duration, 2),
                    kind="instrumental",
                ),
            )

        return timeline
