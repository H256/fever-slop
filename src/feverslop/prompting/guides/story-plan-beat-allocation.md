# Story Plan Beat Allocation

Allocate the story into beats across the supplied segment/shot ids. The
explicit user direction (`creative_direction`) is the highest priority input:
when it conflicts with the story text or the bible, the user direction wins.

Return:

- `beats`: one entry per narrative beat, in narrative order. Each entry has
  `phase` (one of `opening`, `development`, `climax`, `resolution`), a short
  `description`, and the canonical `character_ids`, `location_id`, and
  `prop_ids` that participate in the beat. Reference ONLY the supplied
  canonical ids.
- `brief_allocations`: exactly ONE entry per supplied segment/shot id
  (`target`), each with:
  - `beat_index`: the position (0-based) of the beat this segment belongs to.
  - `required_beat_indices`: beats this segment must also draw from, or empty.
  - `forbidden_beat_indices`: beats this segment must never draw from.

Hard constraints:

- The LAST beat reserves the terminal window: its `phase` is `resolution`.
  No earlier beat may use a terminal phase.
- Every supplied segment/shot id gets exactly one `brief_allocations` entry;
  never invent a new target id.
- `beat_index`, `required_beat_indices`, and `forbidden_beat_indices` are
  positions into the `beats` list (0-based, in range).
- A segment's `beat_index` must be one of its `required_beat_indices` when
  that list is non-empty, and must never be one of its
  `forbidden_beat_indices`.
- Never invent ids, timestamps, frame counts, lyrics, render settings, or
  audio bindings. No timestamps or frame counts of any kind.
