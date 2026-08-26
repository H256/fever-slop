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

## Legacy migration

Existing projects may retain legacy pipeline names and files while they are
being migrated. New projects should persist a concrete `render_profile` ID,
for example `ltx25-i2v-draft`. Migration must resolve old three-field render
settings deterministically from the selected mode and quality; it must never
silently fall back to an older model asset when a 2.5 capability is missing.

Do not delete a legacy project plan until its prompts, frame anchors, audio
timing, and selected workflow have been revalidated under the corresponding
LTX 2.5 profile.
