# Story Plan Beat Allocation

Write a compact story arc as a small sequence of beats. The
explicit user direction (`creative_direction`) is the highest priority input:
when it conflicts with the story text or the bible, the user direction wins.

Return:

- `beats`: one entry per narrative beat, in narrative order. Each entry has
  `phase` (one of `opening`, `development`, `climax`, `resolution`), a short
  `description`, and the canonical `character_ids`, `location_id`, and
  `prop_ids` that participate in the beat. Reference ONLY the supplied
  canonical ids.
Hard constraints:

- The LAST beat reserves the terminal window: its `phase` is `resolution`.
  No earlier beat may use a terminal phase.
- Return only `beats`; do NOT return `brief_allocations`, scene/shot ids, or
  timing information. The application assigns every real scene to your arc
  deterministically, so the end may span several final scenes without asking
  the model to reproduce a large list.
- Never invent ids, timestamps, frame counts, lyrics, render settings, or
  audio bindings. No timestamps or frame counts of any kind.
