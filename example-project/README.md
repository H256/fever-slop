# FeverSlop Quality Benchmark

This tracked project is the stable input fixture for video-profile, prompting,
continuation, and rendering comparisons. Generated plans and media remain
untracked.

## Add the benchmark song

1. Copy the final one-minute MP3 to `input/song.mp3`.
2. Add the complete lyrics to the `lyrics` field in `config.json`.
3. Complete `asset-provenance.json`, including the MP3 SHA-256 and the rights
   information required for repository redistribution.
4. Keep the original MP3 unchanged after its hash has been recorded.

The project intentionally contains no placeholder audio. Until the real song
is added, pipeline commands should fail with a missing-input-audio diagnostic.

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
