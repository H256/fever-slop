# Story Plan Bible

Extract a source-backed, typed story bible from the supplied story text. The
explicit user direction (`creative_direction`) is the highest priority input:
when it conflicts with the story text, the user direction wins.

Return only narrative facts that the supplied sources support:

- `premise`: one or two sentences stating what the story is about.
- `theme`: the emotional or thematic core, in a few words.
- `character_notes`: one short note per supplied character id. Key ONLY the
  supplied canonical character ids; never invent a new id.
- `location_notes`: one short note per supplied location id. Key ONLY the
  supplied canonical location ids.
- `prop_notes`: one short note per supplied prop id. Key ONLY the supplied
  canonical prop ids.
- `narrative_facts`: a list of concrete, source-backed facts (events,
  relationships, stakes) that later planning jobs may rely on.

Hard constraints:

- Reference ONLY the supplied canonical character, location, and prop ids.
  If an entity is not in the supplied lists, do not name it by a new id.
- Never invent ids, timestamps, frame counts, lyrics, render settings
  (resolution, fps, aspect ratio, width, height), or audio bindings
  (segment references, fingerprints, start/end times).
- Do not allocate beats, shots, or segments; that is a later job.
- Keep notes short and factual; no free prose beyond the requested fields.
