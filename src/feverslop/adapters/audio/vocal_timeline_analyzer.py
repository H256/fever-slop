from __future__ import annotations

import gc
import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import librosa
import numpy as np
import torch

from feverslop.domain.timeline import TimelineSegment
from feverslop.domain.vocal_evidence import VocalEvidence, decide_transcript, merge_evidence
from feverslop.ports.reporting import Reporter
from feverslop.utils.sub_step_progress import SubStepProgress
from feverslop.errors import FeverSlopAdaptationError

__all__ = ["VocalTimelineAnalyzer"]


@contextmanager
def _without_incompatible_coverage() -> Iterator[None]:
    """Keep Numba importable when a newer coverage package removed its types API."""
    try:
        coverage = importlib.import_module("coverage")
    except ImportError:
        yield
        return
    if hasattr(coverage, "types"):
        yield
        return

    missing = object()
    original = sys.modules.get("coverage", missing)
    sys.modules["coverage"] = None
    try:
        yield
    finally:
        if original is missing:
            sys.modules.pop("coverage", None)
        else:
            sys.modules["coverage"] = original


def _load_whisper():
    """Import Whisper only when transcription is actually requested."""
    try:
        with _without_incompatible_coverage():
            return importlib.import_module("whisper")
    except (AttributeError, ImportError) as exc:
        raise FeverSlopAdaptationError(
            "Whisper voice analysis could not start. Run `uv sync` to restore the project's "
            "Python environment, then resume. If this project already has a valid vocal timeline, "
            "you can resume with `--skip-whisper`. Technical details: "
            f"{exc}",
        ) from exc


class VocalTimelineAnalyzer:
    def __init__(
        self,
        whisper_model: str = "large-v3",
        language: str = "de",
        merge_gap: float = 0.5,
        min_vocal_duration: float = 0.4,
        min_silence_duration: float = 0.8,
        frame_length: int = 2048,
        hop_length: int = 512,
        rms_low_percentile: float = 20.0,
        rms_high_percentile: float = 85.0,
        rms_ratio: float = 0.35,
        smooth_frames: int = 10,
        reporter: Reporter | None = None,
    ):
        self.reporter = reporter
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

    def set_reporter(self, reporter: Reporter) -> None:
        self.reporter = reporter

    def _message(self, message: str) -> None:
        reporter = getattr(self, "reporter", None)
        if reporter is not None:
            reporter.message(message)

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
        self._message("Transcribing vocal evidence...")
        if self.model is None:
            self.model = _load_whisper().load_model(self.whisper_model)
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

        counts = dict(accepted=0, uncertain=0, rejected=0)
        progress = SubStepProgress(getattr(self, "reporter", None), "Transcript evidence", len(raw_segments))
        progress.update(0, force=True)
        for index, segment in enumerate(raw_segments, 1):
            decision = decide_transcript(segment)
            counts[decision.status] += 1
            if decision.status != "rejected":
                segments.append(segment)
            progress.update(index)
        self._message("Transcript evidence complete: " + ", ".join(f"{key}={value}" for key, value in counts.items()))

        return segments

    def _detect_vocal_activity_rms(
        self,
        vocals_file: Path,
    ) -> tuple[list[tuple[float, float]], float]:
        self._message("Detecting RMS vocal activity...")
        y, sr = librosa.load(str(vocals_file), sr=None, mono=True)

        duration = float(len(y) / sr)

        rms = librosa.feature.rms(
            y=y,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[0]

        if rms.size == 0:
            self._message("RMS activity complete: candidates=0")
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

        progress = SubStepProgress(getattr(self, "reporter", None), "RMS frames", len(active), interval=1000)
        progress.update(0, force=True)
        for i, is_active in enumerate(active):
            progress.update(i + 1)
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

        self._message(f"RMS activity complete: candidates={len(ranges)}")
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
        self._message("Combining transcript and RMS evidence...")
        assigned: list[list[dict]] = [[] for _ in vocal_ranges]
        texts: list[list[str]] = [[] for _ in vocal_ranges]
        evidence: list[VocalEvidence | None] = [None for _ in vocal_ranges]
        outside = []
        progress = SubStepProgress(getattr(self, "reporter", None), "Vocal evidence anchors", len(whisper_segments))
        progress.update(0, force=True)
        for number, ws in enumerate(whisper_segments, 1):
            decision = decide_transcript(ws)
            if decision.status == "rejected":
                progress.update(number)
                continue
            words = ws.get("words") or []
            anchors = words or [{"word": ws["text"], "start": ws["start"], "end": ws["end"]}]
            for word in anchors:
                text = str(word.get("word") or "").strip()
                if decide_transcript(dict(start=word.get("start"), end=word.get("end"), text=text)).status == "rejected":
                    continue
                start, end = float(word["start"]), float(word["end"])
                overlaps = [max(0.0, min(end, stop) - max(start, begin)) for begin, stop in vocal_ranges]
                best = max(overlaps, default=0.0)
                anchor = {"word": text, "start": start, "end": end}
                if best <= 0:
                    outside.append(TimelineSegment(
                        start, end, "vocals", text, (anchor,) if words else (),
                        VocalEvidence("conflict", decision.status, decision.reason_codes + ("transcript_outside_rms",)),
                    ))
                    continue
                index = min(
                    (i for i, overlap in enumerate(overlaps) if overlap == best),
                    key=lambda i: abs((start + end) / 2 - sum(vocal_ranges[i]) / 2),
                )
                if words:
                    assigned[index].append(anchor)
                else:
                    texts[index].append(text)
                partial = start < vocal_ranges[index][0] or end > vocal_ranges[index][1]
                activity = "confirmed" if decision.status == "accepted" else "uncertain"
                anchor_evidence = VocalEvidence(
                    "conflict" if partial else activity,
                    decision.status, decision.reason_codes + (
                        "transcript_crosses_rms_boundary" if partial else "rms_activity",
                    ),
                )
                evidence[index] = (
                    merge_evidence(evidence[index], anchor_evidence)
                    if evidence[index] is not None else anchor_evidence
                )
            progress.update(number)

        result = outside
        for index, (start, end) in enumerate(vocal_ranges):
            words = sorted(assigned[index], key=lambda word: (word["start"], word["end"]))
            text = " ".join([*(word["word"] for word in words), *texts[index]]).strip()
            result.append(TimelineSegment(
                start=float(start), end=float(end), kind="vocals", text=text,
                word_timestamps=tuple(words),
                evidence=evidence[index] or VocalEvidence("uncertain", "missing", ("rms_without_transcript",)),
            ))
        result.sort(key=lambda segment: (segment.start, segment.end))
        conflicts = sum(segment.evidence.activity_status == "conflict" for segment in result)
        uncertain = sum(segment.evidence.activity_status == "uncertain" for segment in result)
        self._message(f"Vocal evidence complete: candidates={len(result)}, uncertain={uncertain}, conflicts={conflicts}")
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
