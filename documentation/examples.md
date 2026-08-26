# Examples

Install dependencies with `uv sync` and configure ComfyUI and the LLM endpoint
in `app_config.json`.

Workflow scene durations are bounded by `video_workflow_limits`, including the
`default_max_render_duration_seconds` setting. Requested scene duration is
clamped to the selected workflow's supported range before rendering.

## Standard project

```bash
uv run python run_pipeline.py ./projects/my-song --skip-tests
```

Use a specific pipeline and selected stages when resuming or debugging:

```bash
uv run python run_pipeline.py ./projects/my-song \
  --video-pipeline minimax-h3-r2v \
  --stage h3_prompts --stage render_plan --skip-tests
```

## Full-Auto project

```bash
uv run python full_auto.py \
  --idea "A cyberpunk chase through a futuristic city" \
  --style "dark synthwave with cinematic drums" \
  --project-name neon-wolves \
  --duration-seconds 120 --width 1280 --height 704 --fps 24 \
  --run-video-pipeline --video-pipeline ltx_msr --skip-tests
```

## Movie project

```bash
uv run python movie_pipeline.py ./projects/my-movie \
  --movie-video-workflow startframe-director
```

## Sequence-based actor and location sheets

Use the opt-in sequence path when references should be derived from a short
multi-view sequence instead of independent image views:

```bash
uv run python run_pipeline.py ./projects/my-song \
  --video-pipeline minimax-h3-r2v \
  --reference-generation sequence_sheet \
  --sequence-to-sheet-workflow workflows/sequence_to_sheet_minimax_h3_i2va_v1.json \
  --stage msr_references --stage msr_reference_sheets \
  --stage h3_prompts --stage render_plan --skip-tests
```

The Movie equivalent is:

```bash
uv run python movie_pipeline.py ./projects/my-movie \
  --movie-video-workflow minimax-h3-r2v \
  --reference-generation sequence_sheet \
  --sequence-to-sheet-workflow workflows/sequence_to_sheet_minimax_h3_i2va_v1.json \
  --skip-movie-render
```

See [the extended tutorial](sequence-reference-pipeline.md) for the artifact
layout, direct `reference_bible` invocation, and review workflow.

## One-machine MiniMax H3 workflow

The tracked `example-project` is the deterministic benchmark baseline. It is
configured for `minimax-h3-r2v` with the draft two-pass profile and contains
the fixed benchmark track plus its lyrics. Use it to compare planner, prompt,
workflow, and render changes without changing the source audio.

When the LLM and ComfyUI cannot keep their models loaded simultaneously, run
the R2V pipeline in four resumable phases. This is also useful when debugging
the hand-off artifacts:

```powershell
# LLM phase: general prompts and model-neutral subject directives
uv run python run_pipeline.py .\projects\my-song `
  --video-pipeline minimax-h3-r2v --stage main_pipeline --skip-tests

# ComfyUI phase: generate actor/location references and sheets
uv run python run_pipeline.py .\projects\my-song `
  --video-pipeline minimax-h3-r2v `
  --stage msr_references --stage msr_reference_sheets --skip-tests

# LLM phase: generate reference-aware H3 prompts
uv run python run_pipeline.py .\projects\my-song `
  --video-pipeline minimax-h3-r2v --stage h3_prompts --skip-tests

# ComfyUI phase: build the plan and render
uv run python run_pipeline.py .\projects\my-song `
  --video-pipeline minimax-h3-r2v `
  --stage render_plan --stage ltx_render_scenes --skip-tests
```

The split is intentional: `main_pipeline` creates the shared scene/action
plan, while `h3_prompts` must wait for the reference sheets so it can use the
actual reference labels and paths.

Every command supports `--help`. Generated artifacts and logs are written
under the selected project's `output/` directory.
