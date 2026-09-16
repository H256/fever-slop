# H3 scene readiness and recovery

H3 preparation records technical `readiness` separately from the artistic judge.
An advisory BAD verdict can render when the technical contract is valid. A
`blocked` scene cannot render or enter final concatenation/muxing. Other scenes
continue preparation before the pipeline reports that preparation is incomplete.

For each scene and relevant input revision, preparation reserves at most one
generation, one corrective model call and one deterministic fallback. Reservations
are saved before work starts under a scene lock. A crash does not restore a used
attempt. The fallback preserves the last structured camera/action plan; it cannot
invent a replacement scene when no usable plan survived. All results, including
manual overrides, must pass the same final contract checks.

### Planner retries reduce flaky hard-blocks

The DSPy planner call is retried automatically (total attempts = `1 + planner_retries`,
default 2) when it returns an unparseable typed plan or throws. This absorbs a
transient bad output from a quantized/verbose serving model instead of hard-blocking
the scene on the first attempt. The repair pass and the deterministic fallback are
unaffected and still run after retries are exhausted. Set `planner_retries=0` to
restore single-attempt behavior.

### Why a scene reports `h3.fallback.plan_missing`

A blocked scene with `h3.fallback.plan_missing` means the DSPy planner produced no
usable structured creative plan at all — it is not an input-validation error. The
most common cause on a local/quantized model is the plan output being truncated
before the typed JSON is complete, which surfaces the plan as missing. The planner's
response token budget is configurable per LLM via `prompt_planner_max_tokens`
(default `H3_PLANNER_MAX_TOKENS = 8192`); raise it if `truncation_suspected` is set
in the diagnostics below. A stronger or more capable serving model is the most
effective fix for persistent recurrence.

### Reading the real failure in a blocked scene

When retries and the deterministic fallback are exhausted, the block record keeps
the concise `h3.fallback.plan_missing` reason in `dspy_error` but now also carries:

- `dspy_error_root`: the underlying planner exception (sanitized of embedded payloads).
- `dspy_error_detail.attempts`: one entry per generation attempt with `cause` and
  an `llm` diagnostic (`budget_max_tokens`, `truncation_suspected`, and the tail of
  the raw model output).

These are persisted with the scene result in `output/render/scenes/scene_NNNN/h3_prompt.json`
so a blocked scene is actionable instead of opaque.

### Opt-in synthesized fallback

`DspyH3PromptBuilder(..., synthesize_plan_fallback=True)` is a last resort that
builds a single-shot, fact-faithful plan from the locked scene facts and base
concept when generation and retries fail, so the scene stays renderable. It is off
by default: the output is deterministic and generic (provenance source
`deterministic_scene_synthesis`), and it still must pass the same final contract
gate — prefer raising the model/token budget or correcting inputs over enabling it.

## Continue or correct a blocked scene

Use the normal safe resume command after correcting the reported timing, lyrics,
cast, reference, workflow or prompt input:

```powershell
uv run python main.py run ./projects/my_song --resume
```

An unchanged blocked scene is reported again without new model calls. Changing
relevant inputs opens a new bounded budget. Changing an output file, random seed
or file modification time does not reset it. Already valid scenes are reused.

To explicitly request another attempt for a selected scene without changing its
inputs, use the existing atomic H3 preparation command:

```powershell
uv run python run_pipeline.py ./projects/my_song --stage h3_prompts --scenes 3
```

This deliberate command opens a new attempt revision for the selected scene,
including a selection covering the whole project. Then run safe resume to finish
the remaining stages. Repeating the explicit command is a new request to spend
another bounded budget; normal `--resume` never does this automatically.

## Diagnostics

Per-scene artifacts are under `output/render/scenes/scene_NNNN/`:

- `h3_prompt.json` retains the result, technical readiness and advisory verdict.
- `h3_prompt.recovery.json` retains input fingerprints, attempt revisions,
  reserved stages and reason codes, including interrupted attempts.

The console identifies blocked scenes, reason codes and reserved attempts.
Correct the inputs or use explicit replanning; deleting recovery artifacts is
not part of normal recovery. Technical validity does not guarantee perceptually
perfect lip sync from the video model.
