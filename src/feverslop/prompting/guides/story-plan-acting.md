# Story Plan Acting

Write structured per-brief acting direction for the supplied segments/shots.
The explicit user direction (`creative_direction`) is the highest priority
input: when it conflicts with the story text, the bible, or the beat
allocation, the user direction wins.

Return:

- `briefs`: one entry per supplied segment/shot id (`target`). Each entry
  carries the service-assigned `beat_id` it belongs to (from the supplied
  beats), plus:
  - `objective`: what the brief must accomplish, in one short sentence.
  - `emotional_turn`: the emotional shift the brief delivers, in a few words.
  - `actor_states`: a list of `{character_id, state}` where `character_id`
    is a supplied canonical character id and `state` is a SHORT state label
    (a phrase, not prose).
  - `character_ids`, `location_id`, `prop_ids`: the canonical entities in
    the brief; reference ONLY the supplied ids.
  - `vocal_presentation`: one of `on_screen`, `offscreen`, `instrumental`.
  - `visual_direction`: a short concrete visual direction.
  - `exclusive`: true only when this brief must own its beat alone.
- `arcs`: one entry per character whose state changes across beats, with the
  service-assigned `beat_ids` it spans and short `from_state` / `to_state`
  labels.

Hard constraints:

- Reference ONLY the supplied beat ids and canonical character, location,
  and prop ids; never invent a new id.
- Structured, not free prose: every field is a short label or sentence,
  never a paragraph of continuity narration.
- Never invent timestamps, frame counts, lyrics, render settings, or audio
  bindings. Do not add `audio_ref`, `id`, or any timing field.
