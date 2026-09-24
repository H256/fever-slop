# Story Plan Beat Allocation

Map the supplied arc skeleton onto the supplied segment/shot ids. The arc
(`beats`) is fixed and already invented; do not add, remove, or reorder
beats. The explicit user direction (`creative_direction`) is the highest
priority input: when it conflicts with the story text or the bible, the
user direction wins.

Return:

- `brief_allocations`: a JSON array with exactly ONE entry per supplied
  segment/shot id. Each entry is an object with:
  - `target`: the supplied segment/shot id (never invent a new one).
  - `beat_index`: the position (0-based) of the beat this segment belongs to.
  - `required_beat_indices`: beats this segment must also draw from, or empty.
  - `forbidden_beat_indices`: beats this segment must never draw from.
  Do not use a JSON object keyed by target; use an array of objects instead.

Hard constraints:

- Every supplied segment/shot id gets exactly one `brief_allocations` entry;
  never invent a new target id.
- `beat_index`, `required_beat_indices`, and `forbidden_beat_indices` are
  positions into the supplied `beats` list (0-based, in range).
- A segment's `beat_index` must be one of its `required_beat_indices` when
  that list is non-empty, and must never be one of its
  `forbidden_beat_indices`.
- Do not invent new beats; reference only the supplied beat positions.
- Never invent ids, timestamps, frame counts, lyrics, render settings, or
  audio bindings. No timestamps or frame counts of any kind.
