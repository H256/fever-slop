# Bounded GPU quality/performance benchmark

`feverslop.tools.gpu_benchmark` renders a declared profile/scene/seed matrix
and records per-cell timing, peak VRAM, and the exact prepared-workflow and
model inventories into a comparable manifest plus a resumable state file.
Unlike manual showcase renders, every cell is driven by an explicit matrix, so
results are reproducible and seed cherry-picking is visible in the manifest.

```bash
uv run python -m feverslop.tools.gpu_benchmark \
  --profiles ltxv_final \
  --scenes 1,2 \
  --seeds 10,20,30 \
  --backend fake \
  --output-dir example-project/output/benchmark
```

The matrix is the Cartesian product of `--profiles`, `--scenes`, and
`--seeds`; all three are required. `--backend fake` renders a deterministic
placeholder (for CI and wiring tests); `--backend comfyui` is reserved for a
configured ComfyUI target.

## Progress and artifacts

The runner emits a `Reporter` step boundary and a `SubStepProgress` update per
cell from the start of the run. Each completed cell is written to
`<output-dir>/state.json` (the resume source of truth) and the final
`<output-dir>/manifest.json` is the comparable review artifact. Re-running the
same matrix skips cells whose recorded output still exists; deleting an output
file forces that cell to re-render.

Contact sheets are opt-in so large renders are not committed by default:
pass `--contact-sheet` to build an ffmpeg montage from the rendered outputs.
Pass `--json` to emit the manifest as machine-readable JSON on stdout.

A tracked, media-free example matrix lives in
`examples/benchmark-music-video/matrix.json`; its rendered outputs are written
under `examples/benchmark-music-video/output/`, which is gitignored.
