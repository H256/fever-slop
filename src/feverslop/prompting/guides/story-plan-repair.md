# Story Plan Repair

You are given the prior typed story plan and a list of structured
diagnostics. Fix ONLY the named diagnostics. Preserve every other field
exactly as it appears in the prior plan.

For each diagnostic:

- `invented_*` / `unknown_segment_target`: replace the offending id with a
  supplied canonical id, or drop the reference when no canonical id fits.
- `beat_index_out_of_range` / `missing_required_beat` /
  `forbidden_beat_allocated`: move the allocation to an in-range beat that
  satisfies the required/forbidden constraints.
- `missing_terminal_window` / `early_terminal_beat`: make the LAST beat the
  terminal window (`resolution`) and remove terminal phases from earlier
  beats.
- `carried_audio_data` / `render_settings_present`: delete the offending
  keys (timestamps, lyrics, render settings) from the brief.
- `missing_audio_ref`: the binding is service-assigned; remove any
  invented audio binding and leave the field to the service.
- `duplicate_id` / `duplicate_brief_target`: keep one entry, drop the
  duplicate.
- `invalid_phase` / `invalid_vocal_presentation`: use a valid enum value.
- `plan_validation_failed`: fix the named structural problem.

Hard constraints:

- Fix ONLY the named diagnostics; do not rewrite unrelated fields.
- Never add ids, timestamps, frame counts, lyrics, render settings, or
  audio bindings.
- Return the complete repaired plan, not a patch.
