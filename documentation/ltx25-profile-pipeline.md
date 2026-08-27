# LTX 2.5 Profile Pipeline

LTX 2.5 is the maintained profile family for new projects. Each mode exposes
`draft`, `standard`, and `final`; all three use the same two-pass topology and
vary only their bounded sampling budget. `draft` is the default for project
creation and benchmark iteration.

| Mode | Profile directory | Primary input | Continuation |
| --- | --- | --- | --- |
| T2V | `workflows/video/ltx_25/t2v/` | prompt | no frame anchor |
| I2V | `workflows/video/ltx_25/i2v/` | start frame + prompt | optional end frame |
| R2V | `workflows/video/ltx_25/r2v/` | predecessor frame + prompt | required predecessor anchor |
| MSR | `workflows/video/ltx_25/msr/` | subject references + prompt | reference-aware |
| Ingredients | `workflows/video/ltx_25/ingredients/` | ingredient references + prompt | reference-aware |

Every generated workflow must use the assets and node classes listed in
`workflows/video/ltx_25/capabilities.json`. The profile sidecar is descriptive
metadata; the JSON workflow remains the executable source and must be validated
against the capability manifest before submission.

## Audio and timing

Native audio is enabled only when the scene declares it. Scene timing remains
absolute against the original song timeline. A continuation segment may begin
with the previous segment's last frame, but it must not duplicate that frame in
the final assembled timeline. The final movie muxes the original song once
after video-only concatenation.

## Prepared-scene continuation scheduling

When the selected prepared LTX profile supports start frames, the renderer
derives continuation predecessors from the visual-consistency contracts. The
prepared-scene stage then runs those scenes through the continuation scheduler:

- a successor waits for its predecessor render and continuity handoff;
- unrelated scenes remain independent and can proceed without waiting for a
  blocked chain;
- changing an upstream scene marks only its downstream handoff suffix dirty;
- each scene and boundary decision is reported through the normal Rich
  progress output, so a long render is observable and resumable.

The scheduler consumes the existing scene workflow manifests and persisted
handoff frame manifests. It does not create a second render-plan format.

Long semantic actions are materialized in the render plan as technical entries
with stable `technical_segment_id` values, deterministic numeric artifact IDs,
absolute `abs_start_seconds`/`abs_end_seconds`, and an explicit
`continuation_predecessor_id`. The semantic scene remains available through
`semantic_scene`, so selecting a scene renders and counts its complete
technical chain. Each technical entry is therefore independently addressable
for workflow preparation, resume, rendering, and diagnostics.

The selected profile's duration capability controls the split. Audio patches
use each entry's absolute timing window, while the predecessor handoff uses the
persisted last-frame boundary manifest. The concat stage discovers the declared
technical segment IDs, validates their boundary evidence, removes only a
proven duplicate first frame, and writes a cutless assembly diagnostics JSON
next to the assembled group. A missing or invalid segment clip blocks that
group from being silently presented as a complete cutless result.

## Legacy migration

Existing projects may retain legacy pipeline names and files while they are
being migrated. New projects should persist a concrete `render_profile` ID,
for example `ltx25-i2v-draft`. Migration must resolve old three-field render
settings deterministically from the selected mode and quality; it must never
silently fall back to an older model asset when a 2.5 capability is missing.

Do not delete a legacy project plan until its prompts, frame anchors, audio
timing, and selected workflow have been revalidated under the corresponding
LTX 2.5 profile.
