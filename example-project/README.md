# FeverSlop Quality Benchmark

This tracked project is the stable input fixture for video-profile, prompting,
continuation, and rendering comparisons. Generated plans and media remain
untracked.

## Benchmark song

The fixed input is `input/the-parts-they-left.mp3`. The complete lyrics are
stored in the `lyrics` field of `config.json`; the musical coverage contract is
stored in `benchmark.json`.

The technical MP3 metadata and SHA-256 are recorded in
`asset-provenance.json`. Complete its remaining rights fields before treating
the binary as release-ready. Keep the MP3 unchanged after its hash has been
recorded.

The deterministic ten-segment planning fixture is `benchmark-plan.json`. It
intentionally contains one continuous action chain so R2V first-/last-frame
handoffs can be tested across the technical duration boundary.

## Run

Preview the planned work before starting expensive stages:

```powershell
uv run python main.py run ./example-project --dry-run
```

After the fixture and planned workflow profiles are complete, resume the
required stages with:

```powershell
uv run python main.py run ./example-project --resume
```
