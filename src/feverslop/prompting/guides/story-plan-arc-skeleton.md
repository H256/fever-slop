# Story Plan Arc Skeleton

Invent the story arc as a coherent whole, independent of the segment count.
The explicit user direction (`creative_direction`) is the highest priority
input: when it conflicts with the story text or the bible, the user
direction wins.

Produce a small set of narrative beats (typically 3-6) that tell one
coherent story from start to finish. This is the spine the rest of the
plan hangs off; it must read as a single arc, not a list of unrelated
moments.

Return:

- `beats`: one entry per narrative beat, in narrative order. Each entry has
  `phase` (one of `opening`, `development`, `climax`, `resolution`), a short
  `description`, and the canonical `character_ids`, `location_id`, and
  `prop_ids` that participate in the beat. Reference ONLY the supplied
  canonical ids.

Hard constraints:

- The FIRST beat has `phase` `opening`; the LAST beat has `phase`
  `resolution`. No other beat may use a terminal phase.
- The arc must be a single coherent progression: setup, rising tension,
  climax, and resolution. Do not produce unrelated or out-of-order beats.
- Keep the beat count small (3-6); do not invent one beat per segment.
- Never invent ids, timestamps, frame counts, lyrics, render settings, or
  audio bindings. No timestamps or frame counts of any kind.
