from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import whisper


@dataclass
class TimelineSegment:
    start: float
    end: float
    kind: str  # "vocals" oder "instrumental"
    text: str = ""


class VocalTimelineAnalyzer:
    def __init__(
        self,
        whisper_model: str = "large",
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
        self.model = whisper.load_model(whisper_model)
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
        )

        segments = []

        for s in result["segments"]:
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
        result = []

        for start, end in vocal_ranges:
            texts = []

            for ws in whisper_segments:
                ws_start = float(ws["start"])
                ws_end = float(ws["end"])

                if ws_start < end and ws_end > start:
                    texts.append(ws["text"].strip())

            result.append(
                TimelineSegment(
                    start=round(float(start), 2),
                    end=round(float(end), 2),
                    kind="vocals",
                    text=" ".join(texts).strip(),
                )
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
                    )
                )

            timeline.append(seg)
            cursor = max(cursor, seg.end)

        if total_duration - cursor >= self.min_silence_duration:
            timeline.append(
                TimelineSegment(
                    start=round(cursor, 2),
                    end=round(total_duration, 2),
                    kind="instrumental",
                )
            )

        return timeline


def normalize_empty_vocals(
    timeline: list[TimelineSegment],
    min_text_chars: int = 3,
) -> list[TimelineSegment]:
    for seg in timeline:
        if seg.kind == "vocals" and len(seg.text.strip()) < min_text_chars:
            seg.kind = "instrumental"
            seg.text = ""
    return timeline


def merge_same_kind_segments(
    timeline: list[TimelineSegment],
    merge_gap: float = 0.5,
) -> list[TimelineSegment]:
    if not timeline:
        return []

    merged = [timeline[0]]

    for seg in timeline[1:]:
        last = merged[-1]

        same_kind = last.kind == seg.kind
        close_enough = seg.start - last.end <= merge_gap

        if same_kind and close_enough:
            last.end = max(last.end, seg.end)

            if seg.text.strip():
                last.text = (last.text + " " + seg.text).strip()
        else:
            merged.append(seg)

    return merged