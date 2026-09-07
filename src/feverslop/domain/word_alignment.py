"""Conservative monotone word alignment; timing is evidence, never interpolation."""
from __future__ import annotations

from copy import deepcopy
from difflib import SequenceMatcher
import hashlib
import json
import math
import re

ALIGNMENT_VERSION = 1


def merge_alignment(left, right):
    """Retain component boundaries and evidence when combining aligned segments."""
    if left.alignment is None and right.alignment is None:
        return None
    components = []
    for segment in (left, right):
        metadata = segment.alignment or align_words(
            segment.text, segment.word_timestamps, segment.text, segment.start, segment.end)
        components.extend(deepcopy(metadata.get("components", [metadata])))
    targets = []
    for component in components:
        offset = len(targets)
        targets.extend(dict(row, target_index=offset + row["target_index"]) for row in component["targets"])
    return {"version": ALIGNMENT_VERSION, "components": components,
            "raw_text": " ".join(c["raw_text"] for c in components).strip(),
            "raw_words": [row for c in components for row in c["raw_words"]],
            "corrected_text": " ".join(c["corrected_text"] for c in components).strip(),
            "targets": targets, "timed_words": [row for row in targets if "start" in row],
            "diagnostics": [row for c in components for row in c["diagnostics"]]}


def normalize_word(value: object) -> str:
    return re.sub(r"[^\w]+", "", str(value).casefold())


def _cost(left: str, right: str) -> int:
    if left and left == right:
        return 0
    # Unrelated substitutions tie with delete+insert and stay unresolved.
    return 1 if left and right and SequenceMatcher(None, left, right).ratio() >= 0.6 else 2


def raw_word_rows(words, *, start: float, end: float) -> list[dict]:
    rows = deepcopy(list(words))
    for index, row in enumerate(rows):
        row.setdefault("source", "legacy_unverified")
        row.setdefault("source_index", index)
        row.setdefault("word_id", f"legacy:{start}:{end}:{index}")
        row.setdefault("segment_start", start)
        row.setdefault("segment_end", end)
    return rows


def alignment_fingerprint(raw_segments, reference: str) -> str:
    encoded = json.dumps([ALIGNMENT_VERSION, raw_segments, reference], sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _timing_diagnostics(rows, start, end):
    diagnostics = []
    invalid = set()
    previous_end = None
    for index, row in enumerate(rows):
        reason = None
        try:
            begin, stop = float(row["start"]), float(row["end"])
        except (KeyError, TypeError, ValueError):
            reason = "missing_timing"
        else:
            if not math.isfinite(begin) or not math.isfinite(stop):
                reason = "nonfinite_timing"
            elif stop <= begin:
                reason = "nonpositive_duration"
            elif begin < max(start, row["segment_start"]) or stop > min(end, row["segment_end"]):
                reason = "out_of_segment"
            elif previous_end is not None and begin < previous_end:
                reason = "nonmonotone_timing"
            if reason is None:
                previous_end = stop
        if reason:
            invalid.add(index)
            diagnostics.append({"source_index": index, "word_id": row["word_id"], "reason": reason})
    return diagnostics, invalid


def align_words(raw_text: str, words, corrected_text: str, start: float, end: float) -> dict:
    """Keep an anchor only if every optimal edit path gives it the same source."""
    rows = raw_word_rows(words, start=start, end=end)
    source = [normalize_word(row.get("word", "")) for row in rows]
    targets = corrected_text.split()
    target = [normalize_word(word) for word in targets]
    n, m = len(source), len(target)
    costs = [[_cost(left, right) for right in target] for left in source]
    forward = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        forward[i][0] = i
    for j in range(m + 1):
        forward[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            forward[i][j] = min(forward[i-1][j] + 1, forward[i][j-1] + 1,
                                forward[i-1][j-1] + costs[i-1][j-1])
    # Traverse the optimal-path DAG backward, collecting all assignments without
    # enumerating its exponentially many complete paths.
    candidates = [set() for _ in target]
    pending = [(n, m)]
    visited = set()
    operations = []
    path = (n, m)
    while pending:
        i, j = pending.pop()
        if (i, j) in visited:
            continue
        visited.add((i, j))
        edges = []
        if i and j and forward[i][j] == forward[i-1][j-1] + costs[i-1][j-1]:
            candidates[j-1].add(i-1)
            edges.append((i-1, j-1, "match" if costs[i-1][j-1] == 0 else "substitution"))
        if j and forward[i][j] == forward[i][j-1] + 1:
            candidates[j-1].add(None)
            edges.append((i, j-1, "insertion"))
        if i and forward[i][j] == forward[i-1][j] + 1:
            edges.append((i-1, j, "deletion"))
        if (i, j) == path and edges:
            left, right, operation = edges[0]
            operations.append({"operation": operation, "source_index": i-1 if i != left else None,
                               "target_index": j-1 if j != right else None})
            path = (left, right)
        # First edge must be visited first for deterministic path reconstruction.
        pending.extend((left, right) for left, right, _ in reversed(edges))
    diagnostics, invalid = _timing_diagnostics(rows, start, end)
    output = []
    timed = []
    for index, word in enumerate(targets):
        choices = candidates[index]
        assigned = next(iter(choices)) if len(choices) == 1 else None
        row = {"word": word, "target_index": index, "source": "unresolved"}
        if len(choices) > 1:
            row["reason"] = "ambiguous"
        elif assigned is None:
            row["reason"] = "unmatched"
        elif assigned in invalid:
            row["reason"] = "invalid_source_timing"
        else:
            anchor = rows[assigned]
            if "raw_segment_index" in anchor:
                row["raw_segment_index"] = anchor["raw_segment_index"]
            row.update({"source_index": anchor["source_index"], "word_id": anchor["word_id"],
                        "raw_word": anchor.get("raw_word", anchor.get("word", "")),
                        "start": anchor["start"], "end": anchor["end"], "source": anchor["source"]})
            if source[assigned] != target[index] and anchor["source"] == "whisper":
                row["source"] = "corrected_from_whisper"
            timed.append(deepcopy(row))
        if row["source"] == "unresolved":
            row["candidate_source_indexes"] = sorted(choice for choice in choices if choice is not None)
        output.append(row)
    for index, row in enumerate(output):
        if row["source"] == "unresolved":
            row["previous_word_id"] = next((r["word_id"] for r in reversed(output[:index]) if "start" in r), None)
            row["next_word_id"] = next((r["word_id"] for r in output[index+1:] if "start" in r), None)
    return {"version": ALIGNMENT_VERSION, "raw_text": raw_text, "raw_words": rows,
            "corrected_text": corrected_text, "targets": output, "timed_words": timed,
            "diagnostics": diagnostics, "operations": list(reversed(operations))}
