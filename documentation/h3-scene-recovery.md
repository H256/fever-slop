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
