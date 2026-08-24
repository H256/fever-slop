# Render Plan Artifacts

Music-video render plans live in `output/render/plans/`. The filenames identify pipeline stages, not schema versions. Do not create chains such as `ingredients_v2_final_fixed.json`.

| File | Owner | Purpose | Recreated by | Retention |
| --- | --- | --- | --- | --- |
| `base.json` | generator (`generated`) and operator (`override`) | Sole editable authoring plan and base resume point | main pipeline, preserving human overrides | Keep while the project may be regenerated |
| `compact.json` | relay compaction | Derived input for anchor correction; never an edit target | relay compact stage | Disposable after `anchored.json` exists |
| `anchored.json` | anchor correction | Derived prompts for storyboard/reference enrichment; never an edit target | anchor fix stage | Keep while downstream plans may be regenerated |
| `references.json` | reference and MSR prompt enrichment | Derived MSR runtime/resume cache; never an edit target | MSR reference/prompt stages | Keep for MSR resume and Ingredients regeneration |
| `ingredients.json` | Ingredients sheet enrichment | Derived Ingredients runtime/resume cache; never an edit target | Ingredients sheets stage | Keep for Ingredients resume; regenerate after upstream prompt/reference edits |

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
overrides by stable scene identity. Supported music-video renderers project
those effective values at their application boundary. They also project
authoritative timing, resolution, seed, and reference-binding fields from
`base.json`; derived reference asset paths remain cache data.

### Backend-specific effective projections

`base.json` remains authority even when the CLI resumes from `anchored.json`,
`references.json`, or `ingredients.json`. Scenes are matched by
`canonical.scene_id`; an identified derived scene that has no match in the
current base plan is rejected instead of being guessed by array position.
Projection changes only a copied runtime/derived scene and never writes an
`effective` value or a derived field back to `base.json`.

| Canonical role | Classic / I2V projection | LTX MSR projection | LTX Ingredients projection | MiniMax H3 projection |
| --- | --- | --- | --- | --- |
| `z_image.prompt` | `z_image.prompt` for storyboard/start frame | Preserved when present | Not part of the compact runtime payload | Preserved when present |
| `ltx.base` | `ltx.base_prompt` fallback | `ltx.base_prompt` fallback | Preserved only until Ingredients-specific global projection | Fallback only when no H3 prompt exists |
| `ltx.i2v` | Both `ltx.i2v_prompt_from_t2i` and `ltx.original_style_i2v_prompt` | Motion fallback | Source context before Ingredients enrichment | Fallback only when no H3 prompt exists |
| `ltx.static` | `ltx.static_prompt` when supported | Unchanged backend-specific payload | Exact V3/V4 static `#PROMPT_POSITIVE` value | Not used |
| `ltx.relay` | `ltx.prompt_relay` | Legacy relay fallback | Legacy relay fallback | Not flattened into H3 text |
| `ltx.msr.global` | Not used | `ltx.msr_global_prompt` / `#PROMPT_RELAY.global_prompt` | Upstream context only | Not used |
| `ltx.msr.relay` | Not used | `ltx.msr_prompt_relay` / local prompt segments | Default source for Ingredients relay generation | Not used |
| `ingredients.global` | Not used | Not used | `ingredients.global_prompt`, mirrored to `ltx.base_prompt` | Not used |
| `ingredients.relay` | Not used | Not used | `ltx.prompt_relay`, retaining segment objects and frame boundaries | Not used |
| `h3.video` | Not used when H3 is not selected | Not used | Not used | `h3.prompt`, passed unchanged to `#PROMPT` after reference-contract validation |
| `performance.timing` | Preserved structured data | Preserved structured data | Preserved when available upstream | `performance_timing`; audio/reference slots remain separate H3 workflow inputs |

MSR and Ingredients therefore continue to have separate global, relay, static,
audio, and reference inputs. They are not concatenated into a universal prompt.
H3 likewise keeps its reference-aware prompt and structured audio/performance
handoffs; changing `h3.video` changes the workflow prompt but does not replace
the reference or audio inputs.

Every projected derived scene contains additive provenance:

```json
{
  "canonical_projection": {
    "schema": "feverslop.canonical-projection/v1",
    "scene_id": "ef679fe8-0157-5c51-af5a-b3affc61ba8a",
    "source": "output/render/plans/base.json",
    "source_revision": "<deterministic SHA-256 of canonical scene data>",
    "dependencies": {
      "schema": "feverslop.canonical-dependencies/v1",
      "source": "output/render/plans/base.json",
      "source_revision": "<same canonical revision>",
      "scene_id": "ef679fe8-0157-5c51-af5a-b3affc61ba8a",
      "workflow_fingerprint": "<normalized per-scene SHA-256>",
      "reference_fingerprint": "<normalized binding SHA-256>"
    }
  }
}
```

`source_revision` identifies the complete canonical plan revision for
diagnostics. It is not a reuse key: changing scene 1 changes the revision but
does not invalidate scene 2. Reuse is decided by the normalized per-scene
fingerprints. `workflow_fingerprint` covers effective prompts and relays,
performance timing, duration/frame settings, resolution, seed, and keyframe
inputs. `reference_fingerprint` covers actor/location and other reference
bindings while excluding generated paths, sheet metadata, and content hashes.
Concrete reference assets and workflow templates are checked separately by the
scene manifest SHA-256 records.

### Invalidation matrix

| Change | Derived reference projection | Prepared workflow | Expensive reference media |
| --- | --- | --- | --- |
| Effective prompt or prompt override | Reused | Affected scene requires `ltx_prepare_workflows` | Reused |
| Performance timing or relay | Reused | Affected scene requires `ltx_prepare_workflows` | Reused |
| Width, height, frame/duration settings | Reused | Affected scene requires `ltx_prepare_workflows` | Reused |
| Seed | Reused | Affected scene requires `ltx_prepare_workflows` | Reused |
| Actor/location reference binding | Stale; rerun `msr_reference_sheets` or `ingredients_sheets` | Prepare only after enrichment | Existing media is retained and reused when the new binding points to it |
| Referenced asset content | Binding remains valid | Manifest asset hash rejects the old workflow; rerun prepare after intentional asset replacement | Only the changed asset must be regenerated/replaced |
| Workflow template content | Reused | Manifest template hash rejects the old workflow; rerun prepare | Reused |
| Unrelated scene edit | Reused | Unchanged scene remains resumable | Reused |

No invalidation step deletes actor/location sheets or other reference media.
It rejects stale use and names the stage that must refresh the projection.

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
canonical/canonical_projection
```

`ltx.prompt_relay` is authoritative for V4 workflows. `ltx.static_prompt` is a deterministic compatibility summary for explicitly selected V3 workflows. V3 remains renderable, but its singing and instrumental transitions are best effort because a single `#PROMPT_POSITIVE` conditioning value cannot enforce frame boundaries.

## Prepared scene artifacts

Each `output/render/scenes/scene_NNNN/` directory contains:

- `h3_prompt.json`: atomically written MiniMax H3 generation checkpoint. It is
  available as soon as that scene finishes its final judge/repair attempt; it
  is generated inspection/resume data, not a human edit target.
- `manifest.json`: schema v3 records the scene-local canonical dependency
  snapshot and hashes the selected workflow template, concrete assets, seed,
  and render settings. V1/v2 manifests remain readable, but canonical-aware
  rendering treats their missing dependency provenance as stale and requires
  prepare.
- `workflow.json`: fully patched ComfyUI API workflow. It is generated output and is overwritten by prepare.
- `raw.mp4`: downloaded ComfyUI result before rolling-window trimming.
- `final.mp4`: scene clip after trimming.

Do not treat `workflow.json` as a planning source. Edit the appropriate plan and rerun prepare.

### MiniMax H3 scene checkpoints

Every completed MiniMax H3 scene writes
`output/render/scenes/scene_NNNN/h3_prompt.json` before generation begins for
the next scene. The checkpoint uses schema
`feverslop.h3-prompt-checkpoint.v1` and records:

- the current scene number plus stable canonical `scene_id` and `segment_id`;
- `generated`, containing the backward-compatible prompt result, references,
  final judge record, and judge attempts;
- status `good`, `bad_exhausted`, or `unjudged`;
- an `input_fingerprint` and non-secret generator provenance.

`bad_exhausted` is a completed checkpoint: every configured repair attempt was
used and the final judge still returned BAD. It remains inspectable and can be
resumed just like GOOD output. Routine console messages show scene, verdict,
status, and path, but never the prompt body.

The fingerprint covers scene/segment identity, concept, scene details, global
context, H3 mode, relay and subject directives, reference/audio asset content,
model and judge configuration, checkpoint contract, and both bundled H3 guide
contents. Reuse requires an exact schema, identity, and fingerprint match.
Changing one scene input therefore regenerates that scene; reordering cannot
attach a checkpoint to a different segment. A malformed checkpoint is an
explicit data error rather than a cache hit.

An interrupted run leaves every already completed checkpoint readable. A
later full run reuses matching checkpoints without an LLM call. `--scenes` is
an explicit regeneration request: selected checkpoints are replaced even when
their fingerprints still match, while unselected checkpoint bytes remain
untouched. At successful batch completion, the legacy
`output/prompts/h3_prompts_<song>.json` list is materialized for existing
readers. A selected run replaces its matching aggregate entries and preserves
the others.

When canonical `base.json` already exists, saving or reusing a checkpoint also
updates only `canonical.roles.h3.video.generated` for the matching stable
identity using an optimistic atomic commit. An existing human `override`
record is never changed. For a new project, `base.json` is created later by the
normal render-plan stage from the compatible aggregate.

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
- Changes to a scene-local workflow dependency or selected workflow invalidate
  only that prepared scene. A complete plan revision alone does not invalidate
  unchanged scenes.

Generated project outputs are local artifacts and must not be committed.

### Read-only CLI reference

Use `main.py plan path PROJECT` to locate the sole editable `base.json`,
`plan validate PROJECT` to validate canonical and legacy provenance, `plan show
PROJECT --scene N` to explicitly inspect generated/override/effective prompt
values, and `plan overrides PROJECT [--orphans]` to audit human ownership.
`main.py status PROJECT` summarizes derived plans, H3 checkpoints, and prepared
workflow freshness without printing prompts. All five commands are read-only;
exit codes are `0` valid/ready, `2` action required, and `1` invalid/corrupt.
Detailed examples are in [`running.md`](running.md#inspecting-canonical-plans-and-artifact-status).

### Supported lower-level legacy inputs

- Plans without `canonical` continue to use their existing `z_image`, `ltx`,
  `h3`, and `performance_timing` fields.
- Application-level storyboard/video requests that omit
  `canonical_plan_path` continue to consume the supplied plan directly.
- Existing filenames, including legacy `render_plan_*`,
  `render_plan_*_refs.json`, and `render_plan_*_ingredients.json`, remain
  readable through the existing artifact lookup.
- MSR relay nodes, Ingredients relay/static workflows, reference images, audio
  slots, and prepared-workflow JSON keep their existing backend-specific
  shapes and anchor names.
- Movie render plans are unchanged and are not projected through this
  music-video canonical contract.

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
