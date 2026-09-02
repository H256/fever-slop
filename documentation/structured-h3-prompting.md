# Structured H3 prompt generation

FeverSlop can generate H3 prompts in two stages:

1. A planner produces only structured sections (`facts`, `shots`, `shot_windows`,
   and optional reference bindings).
2. `DeterministicH3Compiler` sorts, validates, and formats these sections into a
   stable base or full-reference prompt.

The structured path is enabled with `structured_sections` through
`DspyH3PromptBuilder.build_h3_prompt`. For batch processing,
`build_all_h3_prompts` can receive a `structured_sections_by_segment` mapping.
When no mapping is provided, the existing DSPy renderer remains active.

## Contract

- Locked facts require `scene_id`, category, key, value, and `source_id`.
- Each creative shot requires a unique `shot_id`, a visible action, and a
  performance description.
- Every shot must have a valid time window.
- Backend labels and timecodes do not belong in creative fields; they are added
  only during deterministic assembly.
- The output includes `prompt_provenance` with the compiler name and version, so
  later render results can be reproducibly associated with a prompt compiler.

This does not automatically migrate the legacy path. A workflow profile or
planner can explicitly enable the structured path once its sections are
complete.
