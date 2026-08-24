# Render Plan Artifacts

Music-video render plans live in `output/render/plans/`. The filenames identify pipeline stages, not schema versions. Do not create chains such as `ingredients_v2_final_fixed.json`.

| File | Owner | Purpose | Recreated by | Retention |
| --- | --- | --- | --- | --- |
| `base.json` | main render-plan generation | Complete authoring plan and base resume point | main pipeline | Keep while the project may be regenerated |
| `compact.json` | relay compaction | Intermediate input for anchor correction | relay compact stage | Disposable after `anchored.json` exists |
| `anchored.json` | anchor correction | Corrected prompts for storyboard/reference enrichment | anchor fix stage | Keep while downstream plans may be regenerated |
| `references.json` | reference and MSR prompt enrichment | Final MSR render plan and source for Ingredients enrichment | MSR reference/prompt stages | Keep for MSR resume and Ingredients regeneration |
| `ingredients.json` | Ingredients sheet enrichment | Compact runtime plan for Ingredients prepare/render | Ingredients sheets stage | Keep for Ingredients resume; regenerate after upstream prompt/reference edits |

## Canonical scene contract in `base.json`

Newly generated `base.json` scenes contain an additive `canonical` object. The
existing `z_image`, `ltx`, `h3`, and `performance_timing` fields remain in place
for backward compatibility. Older plans without `canonical` remain valid.

```json
{
  "scene": 1,
  "z_image": {"prompt": "legacy-compatible generated prompt"},
  "metadata": {"segment_id": "segment_0001"},
  "canonical": {
    "schema": "feverslop.render-scene.v1",
    "scene_id": "ef679fe8-0157-5c51-af5a-b3affc61ba8a",
    "segment_id": "segment_0001",
    "roles": {
      "z_image.prompt": {
        "generated": {
          "value": "legacy-compatible generated prompt",
          "provenance": {"source": "render-plan-builder"}
        }
      }
    }
  }
}
```

`canonical.scene_id` is deterministically derived from `segment_id`, not from
the scene's array position. Reordering scenes therefore does not change their
identity. Both `scene_id` and `segment_id` must be non-empty and unique within a
plan.

The initial role vocabulary is:

| Role | Meaning |
| --- | --- |
| `z_image.prompt` | Start-frame image prompt |
| `ltx.base` | LTX base prompt |
| `ltx.i2v` | LTX image-to-video prompt |
| `ltx.static` | Static LTX compatibility prompt |
| `ltx.relay` | LTX temporal prompt relay |
| `ltx.msr.global` / `ltx.msr.relay` | MSR global and relay prompts |
| `ingredients.global` / `ingredients.relay` | Ingredients global and relay prompts |
| `h3.video` | MiniMax H3 video prompt |
| `performance.timing` | Structured performance timing |

Roles are deliberately extensible. Consumers must use the shared role resolver
instead of copying precedence rules.

### Human overrides and effective values

Generator-owned and human-owned values are separate. To edit a generated role,
add an `override` sibling; do not replace `generated`:

```json
{
  "generated": {
    "value": "the generated prompt",
    "provenance": {"source": "render-plan-builder"}
  },
  "override": {
    "value": "the operator's prompt",
    "provenance": {"source": "human"}
  }
}
```

Effective values are resolved deterministically in this order:

1. `override.value`, when the override exists and is valid.
2. `generated.value`.
3. An explicit legacy field supplied by a backward-compatible consumer.

Never add an `effective` field to `base.json`. Effective values are computed at
runtime so they cannot become a stale third copy. An override that is present
but empty is an error; remove the whole `override` object to return to the
generated value. Malformed entries report their JSON path, for example
`canonical.roles.h3.video.override.value`.

The following remains a valid legacy scene and resolves through the explicit
legacy fallback until it is migrated:

```json
{
  "scene": 1,
  "z_image": {"prompt": "legacy still prompt"},
  "h3": {"prompt": "legacy H3 prompt"},
  "metadata": {"segment_id": "segment_0001"}
}
```

Existing edits in legacy fields and derived plans can be inspected and migrated
with `plan-migrate`, as described below. Canonical regeneration now preserves
overrides by stable scene identity. Projection into every render backend and
prepared-workflow invalidation remain separate migration steps. Until those
steps land, a stored override is canonical data but is not necessarily consumed
by every renderer.

### Regeneration ownership and conflicts

Render-plan generation owns only `generated`. A regeneration may replace
`generated.value` and its provenance, but it copies each existing `override`
record unchanged to the same `canonical.scene_id` and role. The effective value
therefore remains the human value while the newly generated alternative stays
visible for comparison:

```json
{
  "h3.video": {
    "generated": {
      "value": "new judged H3 prompt",
      "provenance": {"source": "render-plan-builder"}
    },
    "override": {
      "value": "human-approved H3 prompt",
      "provenance": {"source": "human"}
    }
  }
}
```

This generate → override → regenerate contract is also used when H3 generation
is deferred until references exist and when an existing project resumes at the
render-plan stage. Anchor-fix follows the same ownership rule: it updates the
LTX `generated.value` records with `provenance.source = "anchor-fix"`, while
leaving overrides and unrelated roles unchanged.

The writer snapshots the current `base.json` SHA-256 before assembly and checks
it again immediately before one atomic replacement. If the file changes,
disappears, or appears concurrently, regeneration fails instead of overwriting
the newer edit. Rerun regeneration against the new state; do not copy values
from a stale temporary result.

Scene matching never uses array position. Overrides and supported enriched
reference bindings follow `canonical.scene_id`, with `segment_id` checked for
consistency. Reordering is safe. Duplicate identities or a reused `scene_id`
with a different `segment_id` are errors. Deleted/changed identities emit an
`orphaned_override_scene` or `deleted_canonical_scene` diagnostic and are never
reattached by scene number. Unmatched enriched bindings emit
`orphaned_reference_scene`.

With `--scenes`, only selected generated scenes are merged. Every unselected
canonical scene object remains unchanged in `base.json`; the plan is no longer
truncated to the selected subset. A selected regeneration without an existing
canonical base is rejected because there is no safe source for the unselected
scenes. If a selected identity is not regenerated, the old scene is retained
and `selected_identity_missing` is reported.

For MiniMax R2V resume, the identity-based merge preserves only the explicit
reference handoff fields (actor/location sheet and MSR paths, their
descriptions, and `visual_consistency_sources`). Other derived fields are not
copied back into canonical generation. Derived-plan invalidation remains the
responsibility of the later invalidation milestone step.

## Why `ingredients.json` is compact

The Ingredients renderer needs timing and resolution, actor/location IDs, one composed sheet, one stable global prompt, and one temporal prompt relay. It does not need storyboard prompts, full reference manifests, MSR aliases, or every earlier LTX prompt variant.

The runtime scene therefore keeps:

```text
scene/timing/fps/resolution/cut
metadata.segment_id/type/silent_mode/lyrics
references.actor_ids/location_id
ingredients.sheet_path/anchors/global_prompt
ltx.base_prompt/static_prompt/prompt_relay/native_audio
```

`ltx.prompt_relay` is authoritative for V4 workflows. `ltx.static_prompt` is a deterministic compatibility summary for explicitly selected V3 workflows. V3 remains renderable, but its singing and instrumental transitions are best effort because a single `#PROMPT_POSITIVE` conditioning value cannot enforce frame boundaries.

## Prepared scene artifacts

Each `output/render/scenes/scene_NNNN/` directory contains:

- `manifest.json`: hashes the selected workflow template, projected scene, assets, seed, and render settings. Prepare uses it to decide whether cached output is still valid.
- `workflow.json`: fully patched ComfyUI API workflow. It is generated output and is overwritten by prepare.
- `raw.mp4`: downloaded ComfyUI result before rolling-window trimming.
- `final.mp4`: scene clip after trimming.

Do not treat `workflow.json` as a planning source. Edit the appropriate plan and rerun prepare.

## Editing and regeneration

- Treat `base.json` as the only human-editable plan. Put intentional prompt
  changes in `canonical.roles.<role>.override.value`; leave `generated.value`
  intact so generation and human ownership remain distinguishable.
- Treat `compact.json`, `anchored.json`, `references.json`, and
  `ingredients.json` as derived resume/runtime caches. Do not make new edits in
  them. Existing edits can be migrated using the workflow below.
- Deleting `compact.json` after `anchored.json` exists is harmless.
- Deleting `references.json` requires reference/MSR enrichment before MSR or Ingredients preparation.
- Deleting `ingredients.json` requires the Ingredients sheet stage before Ingredients preparation.
- Changes to the projected scene or selected workflow invalidate its prepared manifest and require prepare before render.

Generated project outputs are local artifacts and must not be committed.

## Migrating existing plan edits

Always inspect first. The default command is a write-free dry run:

```bash
uv run python main.py plan-migrate projects/my-song
```

It compares only values for which a defensible baseline exists:

- the legacy-compatible fields in `base.json` against their matching
  `canonical.roles.*.generated.value`;
- pass-through prompt fields in `references.json` (and legacy
  `render_plan_*_refs.json`) against `anchored.json`;
- `ingredients.json[].ltx.prompt_relay` against
  `references.json[].ltx.msr_prompt_relay`.

The supported base roles are `z_image.prompt`, `ltx.base`, `ltx.i2v`,
`ltx.relay`, `h3.video`, and `performance.timing`. Ingredients relay edits are
stored as `ingredients.relay`. Stage-generated MSR values without a historical
baseline are intentionally not guessed.

The tool matches scenes by `canonical.scene_id`, then by a unique `segment_id`,
and only then by a unique legacy scene number. Reordered arrays are therefore
safe. Duplicate identities, orphan scenes, malformed artifacts, missing
baselines, competing candidate values, and conflicts with an existing override
block application and are shown as `UNRESOLVED`.

After reviewing a clean dry run, apply it explicitly:

```bash
uv run python main.py plan-migrate projects/my-song --apply
```

Before changing `base.json`, the command verifies that no analyzed source has
changed, stores byte-identical copies of every analyzed artifact under
`output/render/plans/legacy-migration/<run-id>/`, and writes `report.json` in
the same directory. It then atomically replaces only `base.json`; derived plans
are never rewritten. Repeating the command after a successful import is a
write-free no-op.

Exit status `0` means the dry run or application was clean, `2` means operator
resolution is required, and `1` means the input could not be read or the write
failed. If an apply fails after backup creation, `base.json` remains unchanged;
fix the cause and run the dry run again rather than editing the backup.
