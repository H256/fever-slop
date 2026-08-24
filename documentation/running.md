# Running FeverSlop

This document covers CLI startup and pipeline operation.

## CLI Pipeline

Run an existing standard project:

```bash
uv run python run_pipeline.py ./projects/my-song --skip-tests
```

Ingredients mode (composes per-scene reference sheets from actor/location references):

```bash
uv run python run_pipeline.py ./projects/my-song \
  --video-pipeline ltx_ingredients \
  --skip-tests
```

Passing a config file directly also works:

```bash
uv run python run_pipeline.py ./projects/my-song/config.json --skip-tests
```

Run Full-Auto from the CLI:

```bash
uv run python full_auto.py \
  --idea "A neon chase through a rainy future city" \
  --style "dark synthwave with cinematic drums" \
  --project-name neon-wolves \
  --duration-seconds 120 \
  --width 1280 \
  --height 704 \
  --fps 24
```

Add `--run-video-pipeline` to continue into the normal video pipeline after audio/project creation.

## Safe dry-run and resume

When a pipeline is interrupted or `base.json` changes, first build the immutable
execution plan. Planning reads project artifacts but writes nothing:

```bash
uv run python main.py run ./projects/my-song --dry-run
```

Every phase/scene is reported as:

| Action | Meaning |
| --- | --- |
| `RUN` | Missing or stale work must be recomputed. |
| `REUSE` | The artifact and its stored fingerprints still match. |
| `BLOCKED` | Provenance is unsafe or invalid; run the displayed repair command. |
| `NOT_SELECTED` | The scene is outside the explicit `--scenes` selection. |

Execute the same minimal plan with:

```bash
uv run python main.py run ./projects/my-song --resume
```

For unchanged inputs, dry-run and resume compute the same plan. The resume
command passes only RUN scenes to the existing pipeline and reuses valid scene
artifacts individually. If execution fails, the summary names the last
completed stage and prints the exact safe resume command.

Restrict both planning and execution with ranges:

```bash
uv run python main.py run ./projects/my-song --dry-run --scenes 19-20
uv run python main.py run ./projects/my-song --resume --scenes 19-20
```

Typical decisions are:

- prompt-only edit: RUN projection, preparation, rendering, and assembly for
  that scene; REUSE unrelated scenes;
- reference-binding edit: RUN reference assets/sheets, projection,
  preparation, rendering, and assembly for affected scenes;
- partial H3 batch: REUSE matching judged `h3_prompt.json` checkpoints and RUN
  only missing/stale scenes;
- partial render: REUSE valid `final.mp4` clips and RUN missing scenes plus
  assembly;
- stale workflow or changed timing/resolution/template: RUN preparation before
  rendering, so a stale prepared workflow is never submitted silently;
- ambiguous legacy edit or malformed override: BLOCKED with `plan-migrate` or
  `plan validate`; no pipeline stage starts.

### Advanced compatibility commands

Atomic stages and skip flags remain supported for diagnostics, forced reruns,
and scripts. They are translated into the same typed plan when used through
`main.py run`, but they deliberately bypass normal minimal-resume selection.
For example:

```bash
uv run python main.py run ./projects/my-song --resume \
  --stage ltx_render_scenes \
  --scenes 19-20
```

The original entry point is unchanged:

```bash
uv run python run_pipeline.py ./projects/my-song \
  --stage ltx_render_scenes \
  --no-skip-existing
```

Compatibility skip flags include:

| Flag | Skips |
| --- | --- |
| `--skip-main-pipeline` | Timeline, prompts, render plan generation |
| `--skip-msr-reference-render` | MSR reference sheet rendering |
| `--skip-msr-prompt-enrichment` | LLM-based MSR prompt enrichment |
| `--skip-ingredients-sheets` | Ingredients scene sheet composition |
| `--skip-ltx` | Entire video rendering stage |
| `--skip-final-concat` | Concatenation and audio muxing |

Prefer the normal dry-run/resume pair unless you intentionally need an atomic
stage or forced regeneration.

## Recovering after editing `base.json`

The CLI verifies each prepared scene against the current authoritative
`output/render/plans/base.json` before sending a workflow to ComfyUI. A changed
prompt, timing, resolution, seed, or relay produces an error like:

```text
Stale prepared workflow from output/render/plans/base.json for scene 3: workflow fingerprint changed. Run --stage ltx_prepare_workflows first.
```

The normal recovery command derives the affected scenes and prerequisite stages:

```bash
uv run python main.py run ./projects/my-song --dry-run
uv run python main.py run ./projects/my-song --resume
```

Other scenes remain resumable when
their own dependency fingerprints and artifacts are unchanged. A changed
overall canonical revision does not by itself force every scene to prepare.

Changing actor/location IDs or another reference binding first reports, for
example:

```text
Stale derived reference binding from output/render/plans/base.json for scene 3: reference binding fingerprint changed. Run --stage ingredients_sheets first.
```

Normal resume selects the appropriate Ingredients or MSR reference stages and
then preparation automatically. These checks do not delete existing reference
media. Actor/location sheets are reused when the new binding still selects the
same files; only a changed/missing reference asset needs regeneration.

Existing v1/v2 scene manifests and manifests without canonical dependency
provenance remain readable for legacy tooling. When a current canonical scene
is supplied, rendering fails closed with `canonical dependency provenance is
missing`; running `ltx_prepare_workflows` upgrades that scene to a v3 manifest.

An intentional change to a referenced file or workflow template is detected by
its stored SHA-256 even when `base.json` is unchanged. Rerun prepare after the
asset/template is in its intended final state; never edit generated
`workflow.json` directly.

## Inspecting canonical plans and artifact status

All inspection commands are read-only. They do not regenerate plans, update
checkpoints, prepare workflows, or create migration backups:

```bash
uv run python main.py plan path projects/my-song
uv run python main.py plan validate projects/my-song
uv run python main.py plan show projects/my-song --scene 3
uv run python main.py plan overrides projects/my-song
uv run python main.py plan overrides projects/my-song --orphans
uv run python main.py status projects/my-song
```

`plan path` labels `output/render/plans/base.json` as the sole editable plan
and the other plan files as derived caches. `plan show` is the explicit prompt
inspection command: it prints generated, override, effective value, owner, and
provenance for the selected scene. `plan overrides` lists override ownership;
`--orphans` additionally shows unmatched legacy/derived records.

`plan validate` checks canonical identities and role contracts, override and
provenance structure, duplicate/orphaned identities, malformed artifacts, and
unresolved legacy migration findings. `status` reports phase/scene states
`READY`, `STALE`, `PARTIAL`, `BLOCKED`, and `MISSING`, including canonical
revision, derived fingerprints, H3 judge/fingerprint state, prepared workflow
freshness, cause, and required next phase. Routine status never prints prompt
bodies.

Exit codes are stable across these commands:

| Code | Meaning |
| --- | --- |
| `0` | The requested inspection is valid; status has no required action. |
| `2` | The project is readable but stale, partial, missing an artifact, or otherwise needs an operator action. |
| `1` | The project or inspected artifact is invalid/corrupt. |

Typical output includes `PARTIAL ... required next phase: h3_prompts`,
`STALE ... workflow fingerprint changed; required next phase:
ltx_prepare_workflows`, or `BLOCKED ... required next phase: plan-migrate`.
These are observations only; construction and execution of a multi-stage
resume plan are separate commands introduced after this inspection layer.

Representative inspections:

```text
# generated-only / overridden roles (only plan show exposes values)
z_image.prompt | generated | <generated value> | —                | <generated value>
z_image.prompt | override  | <generated value> | <override value> | <override value>

# unmatched legacy record
9 | unmatched | - | ORPHAN output/render/plans/references.json: orphan scene

# routine status (never includes the values above)
h3 checkpoint    | 3 | PARTIAL | missing; required next phase: h3_prompts
derived plan     | 3 | STALE   | reference fingerprint changed; required next phase: ingredients_sheets or msr_reference_sheets
prepared workflow| 3 | STALE   | workflow fingerprint changed; required next phase: ltx_prepare_workflows
```

## Inspecting and resuming MiniMax H3 prompts

Each judged scene becomes inspectable immediately, without waiting for the
complete batch:

```text
projects/<project>/output/render/scenes/scene_NNNN/h3_prompt.json
```

This is generated checkpoint/diagnostic data. Do not edit it. Status `good`
means the final judge accepted the prompt; `bad_exhausted` means all repair
attempts completed but the final verdict remained BAD; `unjudged` records a
completed result without a judge. Console progress prints the scene, verdict,
status, and path without printing the prompt body.

After an interruption, inspect and resume H3 generation from its per-scene
checkpoints with:

```powershell
uv run python main.py run .\projects\my-song --dry-run
uv run python main.py run .\projects\my-song --resume
```

Only checkpoints whose complete input fingerprints still match are reused.
A changed concept, scene direction, relay, subject directive, reference/audio
asset, H3 guide, model, or judge configuration regenerates the affected scene.
To plan only selected scenes, append `--scenes 2,5-6`. Matching checkpoints in
the selection are still reused; use the advanced atomic stage command only
when deliberate forced generation is required. Other scene checkpoint files
are not rewritten.

The compatibility aggregate remains at
`output/prompts/h3_prompts_<song>.json` and is rebuilt after a successful H3
batch. It is also generated data, not the edit target. Human corrections belong
only in `output/render/plans/base.json` under
`canonical.roles.h3.video.override.value`. Checkpoint save/reuse may update the
matching `generated` value, but never that override. Backend consumption of the
effective override is introduced by the subsequent canonical projection step.

## Migrating edits from existing render plans

Older projects may contain manual edits in `base.json` legacy fields,
`references.json`, `ingredients.json`, or their old
`output/render/render_plan_*.json` counterparts. Preview evidence-backed
imports into canonical `base.json` overrides with:

```bash
uv run python main.py plan-migrate projects/my-song
```

A successful preview reports the number of `importable` values and ends with
`Dry run complete; no files were written`. Review the listed scene identity,
role, field, source path, and reason. Prompt bodies are deliberately not printed.
Apply only after the preview has no `UNRESOLVED` findings:

```bash
uv run python main.py plan-migrate projects/my-song --apply
```

Application first creates a timestamped backup below
`output/render/plans/legacy-migration/`, then atomically updates only
`output/render/plans/base.json`. Derived plans remain untouched.

A blocked example looks like this:

```text
Found 0 importable, 1 unresolved, and 0 already applied value(s).
  UNRESOLVED | ... | z_image.prompt | conflicting candidate values
Blocked: resolve every unresolved finding before applying.
```

The command exits with status `2` in that case. Resolve the conflicting source
edits or the existing canonical override, rerun the dry run, and apply only when
the evidence is unambiguous. See
[`render-plan-artifacts.md`](render-plan-artifacts.md#migrating-existing-plan-edits)
for supported roles, matching rules, proof limits, backup contents, and failure
recovery.

FFmpeg, ComfyUI, and the configured LLM endpoint remain external pipeline requirements. Their addresses are defined by project and application configuration.
LLM prompt-generation calls share the process-local
`llm.max_concurrent_requests` budget from `app_config.json`; the default is
`1`, which keeps direct OpenAI-compatible calls and DSPy/LiteLLM calls from
running concurrently inside one FeverSlop process. This does not limit other
FeverSlop processes or other clients hitting the same LLM server.
