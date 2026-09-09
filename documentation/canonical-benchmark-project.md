# Canonical benchmark project

`example-project/` is the tracked, media-backed input for the one-minute
benchmark. Its `benchmark.json` manifest is versioned separately from pipeline
outputs and records the Python/uv/toolchain contract, the portable project
paths, and the SHA-256 of every binary input. A clean checkout can validate it
without any local `projects/` state:

```bash
uv run python -m feverslop.tools.benchmark_fixture example-project
```

The expected planning target lives in
`tests/fixtures/canonical_benchmark/benchmark_plan.json`; it is not generated
output. Runs write to `example-project/output/`, which is ignored by the
fixture's `.gitignore` (as are `expected-output/`, `.studio/`, and other local
state). Do not add rendered media or machine-specific absolute paths to the
tracked fixture.
