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

## Resuming a failed pipeline

When a pipeline fails mid-render (e.g., ComfyUI timeout, network interruption), the already-rendered scene files remain on disk. You can resume from the point of failure instead of re-rendering everything.

**Resume rendering remaining scenes:**

```bash
uv run python run_pipeline.py ./projects/my-song \
  --skip-tests \
  --skip-main-pipeline \
  --skip-msr-reference-render \
  --skip-msr-prompt-enrichment \
  --stage ltx_render_scenes \
  --stage concat_video_only \
  --stage mux_original_audio
```

This skips all pre-processing stages and goes straight to rendering. Existing scene files are detected automatically (default `skip_existing=true`), so only missing scenes are re-rendered.

**Render only specific scenes:**

Use `--scenes` to target individual scene numbers (comma-separated, ranges supported):

```bash
uv run python run_pipeline.py ./projects/my-song \
  --stage ltx_render_scenes \
  --scenes 19-20
```

**Force re-rendering all scenes:**

Add `--no-skip-existing` to ignore existing scene files and re-render everything:

```bash
uv run python run_pipeline.py ./projects/my-song \
  --stage ltx_render_scenes \
  --no-skip-existing
```

**Typical skip flags for resuming:**

| Flag | Skips |
| --- | --- |
| `--skip-main-pipeline` | Timeline, prompts, render plan generation |
| `--skip-msr-reference-render` | MSR reference sheet rendering |
| `--skip-msr-prompt-enrichment` | LLM-based MSR prompt enrichment |
| `--skip-ingredients-sheets` | Ingredients scene sheet composition |
| `--skip-ltx` | Entire video rendering stage |
| `--skip-final-concat` | Concatenation and audio muxing |

Combine `--skip-*` flags with `--stage` to run only the stages you need.

## Editing MiniMax H3 prompts after generation

For MiniMax H3 projects, the generated prompts are stored in:

```text
projects/<project>/output/prompts/h3_prompts_<song>.json
```

You can edit the `prompt` value of an individual `segment_id` directly in
this JSON file, for example to correct an action or add a directing detail.
The renderer does not read this file directly. First copy the edited prompts
into the render plan by running `render_plan`, then render the scenes:

```powershell
uv run python run_pipeline.py .\projects\my-song `
  --video-pipeline minimax-h3-r2v `
  --skip-tests `
  --skip-main-pipeline `
  --skip-msr-reference-render `
  --skip-msr-prompt-enrichment `
  --stage render_plan `
  --stage ltx_render_scenes `
  --stage concat_video_only `
  --stage mux_original_audio
```

The effective render source after this step is
`output/render/plans/base.json`; ComfyUI receives the prompt from that plan.
Keep the existing `references` data intact, especially for `r2v`, and change
only the prompt text unless you intentionally want to alter reference or
timing data. If only selected scenes should be regenerated, add
`--scenes 2,5-6` to the command.

These edits are manual overrides, not a second prompt-generation layer.
Running the H3 prompt-generation stage again regenerates
`h3_prompts_<song>.json` and can overwrite the manual changes. Save or copy
your edited file before regenerating prompts.

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
