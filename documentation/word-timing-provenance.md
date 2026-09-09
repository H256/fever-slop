# Word timing after lyric correction

Lyric correction changes text without stretching word timings across a vocal
segment. A monotone edit alignment preserves unique exact word anchors, ignoring
case and punctuation. A unique lexical substitution (normalized character
similarity at least 0.6) can inherit its original interval. Insertions, unrelated
words, and repeated words with multiple equally good assignments remain
unresolved. No intervals are interpolated.

Fresh analyzer words carry `source: whisper`, stable `word_id`,
`raw_segment_index`, and `source_index`. Corrected fresh words carry
`corrected_from_whisper`; timings from older, unmarked documents remain
`legacy_unverified`, including after correction. This records provenance, not
proof that a transcription is correct.

Timeline JSON retains its existing text and `word_timestamps` fields. The
additive `alignment` object stores raw text and words, corrected text, all target
positions, edit operations, and diagnostics. Unresolved targets have a reason,
candidate source positions, and neighboring timed word IDs, but no fabricated
start or end. The timed subset remains available in `word_timestamps`.

Missing, nonfinite, nonpositive, nonmonotone, and out-of-segment intervals are
diagnosed and excluded from the timed subset. Raw values are retained without
clamping. Original component bounds survive a merge, so a wider merged segment
cannot silently validate a previously out-of-bounds anchor. Merged alignment
metadata retains its component records.

A fingerprint of raw segments, normalized reference lyrics, and alignment
version skips the LLM for unchanged resumes. Changed lyrics are sent alongside
raw transcript text and aligned against raw word rows, never against an earlier
correction. Serialization retains full timing precision. Pipeline reporting
shows segment progress and timed, unresolved, and invalid counts without logging
lyrics or transcript payloads.

Downstream consumers must inspect provenance and unresolved targets when they
need reliable coverage. A corrected lyric string does not imply every word has
supported timing.

## Stable timeline references

Performance projections now carry an additive reference to their source
timeline segment. `timeline_indices` and `alignment_refs` on each phase, plus
`timeline_index` and `alignment_ref` on each vocal event, identify the owning
entry in `timeline_song.json` without copying that entry's identity into
consumer-specific schemas. References use the JSON-safe shape
`{"version": 1, "timeline_index": N}` and are stable when a scene is
re-projected with different bounds.

This is the first compatibility step toward a lean projection model. Existing
`vocal_sources`, `vocal_events`, and word timing fields remain available so
current stage1, relay, render, and H3 consumers produce the same output. A
follow-up migration can resolve source evidence from these references and then
remove the duplicated source payloads once all consumers have switched.
