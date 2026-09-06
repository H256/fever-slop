# Release gates for video profile changes

Profile changes are released only after every applicable gate below passes.
The gates separate structural correctness from visual quality and machine
performance; one attractive render is not evidence that a default is safe.

## Gate 1: CPU golden checks

Run the complete CPU-test contract from a clean checkout:

```bash
uv run python -m tests.suites.unit
```

The result must be successful, and the checked-in example project plus its
canonical prompt, render-plan, prepared-workflow, and resume artifacts must
remain loadable. A failed or missing golden assertion blocks the release.

## Gate 2: profile and workflow preflight

Resolve each supported final profile before contacting ComfyUI:

```bash
uv run feverslop profiles preflight --app-config app_config.json --pipeline ltx_i2v --purpose final
uv run feverslop profiles preflight --app-config app_config.json --pipeline minimax-h3-r2v --purpose final
uv run feverslop profiles preflight --app-config app_config.json --pipeline minimax-h3-t2v --purpose final
```

Each command must resolve the intended profile and workflow. Validate the
workflow JSON and capability manifest as part of the corresponding focused
tests; missing model assets, anchors, or node classes are release blockers.

## Gate 3: real smoke renders

Run one short real render for every supported mode on the target ComfyUI
installation: LTX 2.5 T2V, I2V, R2V, MSR, Ingredients, and MiniMax H3 modes
that are advertised by the release. Use the checked-in example project and
record the command, commit, workflow profile, GPU, driver, duration, and
output artifact. A smoke render must complete with playable audio/video and
without a missing-node or missing-model error.

The command shape is:

```bash
uv run python run_pipeline.py ./example_movie_project \
  --run-video-pipeline --video-pipeline minimax-h3-r2v --skip-tests
```

Repeat it with the selected LTX 2.5 or H3 mode and the release's documented
workflow configuration; do not treat `--skip-tests` as skipping this gate.

This gate requires the actual render environment; CPU tests or a workflow
parse alone do not substitute for it.

## Gate 4: three-seed quality review

For every calibrated profile, render the same representative scenes with
three recorded seeds. Review the outputs side by side for prompt adherence,
identity/reference consistency, temporal stability, audio timing, and visible
artifacts. Store the seed list, profile, scene IDs, and reviewer decision with
the benchmark record. A profile passes only when all three seeds meet the
documented quality bar; do not accept a default based on one favorable seed.

## Gate 5: bounded performance review

Record wall time, peak VRAM, output resolution, frame count, and failures for
the smoke and three-seed runs. Compare them with the previous default. The
release record must state the accepted runtime/VRAM regression budget and list
any unresolved limitation. An unexplained regression or missing measurement
blocks the default change.

## Gate 6: ordered migration gate

LTX 2.3 removal is evaluated only after Gates 1–5 pass for LTX 2.5. Confirm
that new projects resolve LTX 2.5 profiles, existing projects migrate or
remain explicitly supported, and no active command or workflow silently points
at LTX 2.3. If any LTX 2.5 gate fails, keep the LTX 2.3 compatibility path and
record the failure instead of removing it.

## Release record

The release record must link the test output, preflight output, smoke-render
artifacts, three-seed review, performance measurements, and migration result.
It must identify the commit and reviewer. Unavailable GPU or model assets are
an explicit `blocked` result, not a pass.
