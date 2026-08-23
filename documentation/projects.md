# Projects and Configuration

FeverSlop projects are ordinary directories under `projects/`. CLI commands
accept either a project directory or its `config.json` path.

```text
projects/
`-- my-song/
    |-- config.json
    |-- input/
    |   `-- my-song.mp3
    `-- output/
```

## Standard projects

Create `config.json` with at least a project name and input audio, then run:

```bash
uv run python run_pipeline.py ./projects/my-song --skip-tests
```

The configuration contains the audio/video settings, visual direction, and
selected pipeline. See `config/config.example.json` for the complete shape.

## Full-Auto projects

`full_auto.py` creates the project artifacts from an idea and style. Add
`--run-video-pipeline` to continue directly into rendering:

```bash
uv run python full_auto.py \
  --idea "A cyberpunk chase through a futuristic city" \
  --style "dark synthwave with cinematic drums" \
  --project-name neon-wolves \
  --duration-seconds 120 --width 1280 --height 704 --fps 24 \
  --run-video-pipeline --video-pipeline ltx_msr --skip-tests
```

## Movie projects

Use `movie_pipeline.py` for screenplay/movie projects. It writes the movie
Bible, render plan, references, and MSR-enrichment artifacts into the project
directory and supports the same `--scenes` and workflow options documented by
the command's `--help` output.

## Inspecting artifacts

All generated JSON, images, scene clips, and final videos remain in the
project's `output/` directory. They are intentionally plain files so they can
be inspected, versioned selectively, or processed by other CLI tools.
