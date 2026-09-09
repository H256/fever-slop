"""Project vocal evidence onto scene windows without inventing measured activity."""
from __future__ import annotations

from copy import deepcopy
import math


PERFORMANCE_TIMELINE_VERSION = 1
ALIGNMENT_REFERENCE_VERSION = 1
_SUPPORTED = {"whisper", "corrected_from_whisper"}


def alignment_reference(timeline_index: int) -> dict:
    """Return the stable, JSON-safe reference to a source timeline segment."""
    return {"version": ALIGNMENT_REFERENCE_VERSION, "timeline_index": timeline_index}


def _bounds(row):
    try:
        start, end = float(row["start"]), float(row["end"])
    except (KeyError, TypeError, ValueError):
        return None
    return (start, end) if math.isfinite(start) and math.isfinite(end) and end > start else None


def project_performance(timeline: list[dict], start: float, end: float) -> list[dict]:
    """Return absolute half-open phases, retaining source evidence and clipped words.

    Word occupancy determines performance; midpoint ownership determines text only.
    Legacy envelopes remain readable but explicitly lack acoustic verification.
    Explicit instrumental entries and vocal envelopes cover only their own bounds;
    uncovered timeline gaps never count as evidence of silence.
    """
    if end <= start:
        return []
    sources = []
    coverage = []
    cuts = {start, end}
    for index, segment in enumerate(timeline):
        bounds = _bounds(segment)
        if bounds is None or bounds[1] <= start or bounds[0] >= end:
            continue
        kind = segment.get("type") or segment.get("kind")
        if kind not in {"vocals", "instrumental"}:
            continue
        covered_start, covered_end = max(start, bounds[0]), min(end, bounds[1])
        cuts.update((covered_start, covered_end))
        if kind == "instrumental":
            evidence = segment.get("evidence")
            reasons = (["uncertain_vocal_evidence"]
                       if evidence and evidence.get("activity_status") != "confirmed" else [])
            coverage.append((covered_start, covered_end, reasons))
            continue
        alignment = segment.get("alignment") or {}
        words = alignment.get("timed_words", segment.get("word_timestamps") or [])
        valid = []
        for word_index, word in enumerate(words):
            word_bounds = _bounds(word)
            if word_bounds is None:
                continue
            row = deepcopy(word)
            row.setdefault("word_id", f"legacy:{bounds[0]}:{bounds[1]}:{word_index}")
            row.setdefault("source", "legacy_unverified")
            valid.append(row)
        reasons = []
        if not valid or any(word.get("source") not in _SUPPORTED for word in valid):
            reasons.append("legacy_timing_unverified")
        if any(row.get("source") == "unresolved" for row in alignment.get("targets", [])):
            reasons.append("unresolved_lyric_alignment")
        if len(valid) != len(words):
            reasons.append("invalid_word_timing")
        evidence = segment.get("evidence")
        if evidence and evidence.get("activity_status") != "confirmed":
            reasons.append("uncertain_vocal_evidence")
        # Distinct voices can start and stop inside a shared vocal envelope.
        voice_intervals = {}
        for word in valid:
            identity = tuple(word.get(key, segment.get(key)) for key in
                             ("subject_id", "subject_label", "speaker_id", "offscreen"))
            voice_intervals.setdefault(identity, []).append((
                max(start, bounds[0], float(word["start"])),
                min(end, bounds[1], float(word["end"])),
            ))
        for intervals_for_voice in voice_intervals.values():
            merged_voice = []
            for left, right in sorted(intervals_for_voice):
                if right <= left:
                    continue
                if merged_voice and left <= merged_voice[-1][1]:
                    merged_voice[-1] = (merged_voice[-1][0], max(right, merged_voice[-1][1]))
                else:
                    merged_voice.append((left, right))
            for left, right in merged_voice:
                cuts.update((left, right))
        coverage.append((covered_start, covered_end, reasons))
        intervals = [(max(bounds[0], float(w["start"])), min(bounds[1], float(w["end"]))) for w in valid]
        if not valid:
            intervals = [bounds]
        intervals = [(max(start, a), min(end, b)) for a, b in intervals if b > start and a < end and b > a]
        merged = []
        for a, b in sorted(intervals):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
            else:
                merged.append((a, b))
        intervals = merged
        for a, b in intervals:
            cuts.update((a, b))
        sources.append((index, segment, valid, intervals, reasons))
    phases = []
    ordered = sorted(cuts)
    for a, b in zip(ordered, ordered[1:]):
        active = [(i, s, words, reasons) for i, s, words, intervals, reasons in sources
                  if any(left < b and right > a for left, right in intervals)]
        clipped, vocal_sources, lyrics, events = [], [], [], []
        timeline_indices = []
        for index, segment, words, reasons in active:
            timeline_indices.append(index)
            event_words, event_lyrics = [], []
            source = deepcopy(segment)
            source["timeline_index"] = index
            vocal_sources.append(source)
            if words:
                for word in words:
                    left, right = float(word["start"]), float(word["end"])
                    if left >= b or right <= a:
                        continue
                    row = deepcopy(word)
                    row.update(start=max(a, left), end=min(b, right))
                    row["source_start"] = left
                    row["source_end"] = right
                    if left < a or right > b:
                        row["continuation_of_word"] = row["word_id"]
                    row["text_assigned"] = a <= (left + right) / 2 < b
                    if row["text_assigned"] and str(row.get("word") or "").strip():
                        lyrics.append(str(row["word"]).strip())
                        event_lyrics.append(str(row["word"]).strip())
                    clipped.append(row)
                    event_words.append(deepcopy(row))
            else:
                # Legacy text is owned by one phase; its timing remains unverified.
                midpoint = (float(segment["start"]) + float(segment["end"])) / 2
                if a <= midpoint < b:
                    text = str(segment.get("lyrics") or segment.get("text") or "").strip()
                    lyrics.append(text)
                    event_lyrics.append(text)
            identities = ("subject_id", "subject_label", "speaker_id", "offscreen")
            groups = {}
            for word in event_words:
                identity = tuple(word.get(key, segment.get(key)) for key in identities)
                groups.setdefault(identity, []).append(word)
            if not groups:
                groups[tuple(segment.get(key) for key in identities)] = []
            for identity, selected_words in groups.items():
                selected_text = " ".join(str(word.get("word") or "").strip() for word in selected_words if word["text_assigned"])
                if not selected_words:
                    selected_text = " ".join(event_lyrics)
                event = {**deepcopy(segment), "start": a, "end": b,
                         "lyrics": selected_text, "text": selected_text,
                         "word_timestamps": selected_words,
                         "timeline_index": index,
                         "alignment_ref": alignment_reference(index)}
                event.update({key: value for key, value in zip(identities, identity) if value is not None})
                events.append(event)
        covering = [reasons for left, right, reasons in coverage if left <= a and right >= b]
        reasons = list(dict.fromkeys(reason for source_reasons in covering for reason in source_reasons))
        if not covering:
            reasons.append("missing_performance_evidence")
        phases.append({"start": a, "end": b, "state": "singing" if active else "instrumental",
                       "lyrics": " ".join(lyrics), "word_timestamps": clipped,
                       "vocal_sources": vocal_sources, "vocal_events": events, "performance_phase": True,
                       "timeline_indices": timeline_indices,
                       "alignment_refs": [alignment_reference(index)
                                          for index in timeline_indices],
                       "performance_intervals_version": PERFORMANCE_TIMELINE_VERSION,
                       "acoustically_verified": not reasons,
                       "reason_codes": reasons,
                       "performance_conflicts": [{"reason_code": reason} for reason in reasons]})
        if len(events) == 1:
            for key in ("subject_id", "subject_label", "speaker_id", "offscreen"):
                if key in events[0]:
                    phases[-1][key] = deepcopy(events[0][key])
    return phases
