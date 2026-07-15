# FeverSlop

FeverSlop is a local music-video generation pipeline and Studio UI. It turns an audio track, lyrics, and visual direction into project artifacts, render plans, ComfyUI image/video jobs, reviewable clips, and a final muxed video.

The repository contains two operator surfaces:

- **CLI pipeline**: `run_pipeline.py` and `full_auto.py`.
- **FeverSlop Studio**: FastAPI backend plus Vue frontend for project creation, config editing, pipeline jobs, logs, references, review, and final video preview/download.

Core Python packages live under `src/feverslop`. Composition code such as
`feverslop.composition.generate_render_plan` is kept separate from adapters
such as `feverslop.adapters.comfyui_video_backend`.

## Quick Start

Install Python dependencies from the repository root:

```bash
uv sync
```

Install frontend dependencies:

```bash
cd studio/frontend
bun install
```

Start Studio backend:

```bash
uv run python -m feverslop.studio.server
```

Start Studio frontend in another terminal:

```bash
cd studio/frontend
bun run dev
```

Open:

```text
http://127.0.0.1:5173
```

The Vite dev server proxies `/api` to the FastAPI backend on `http://127.0.0.1:8765`.

## Basic Workflows

### Standard Music Video Project

1. Open Studio.
2. Click **Create Project**.
3. Choose **Standard - Music Video Project**.
4. Enter a project name. Studio slugifies it into the project folder name under `projects/`.
5. Open **Project Settings** and fill `config.json` fields, especially `input_audio`.
6. Choose `video_pipeline`:
   - `ltx_msr`: MSR/reference-guided mode.
   - `ltx_i2v`: classic storyboard/start-frame mode.
   - `ltx_ingredients`: per-scene ingredients sheets with audio latent injection.
7. Open **Pipeline**, start a job, and monitor progress/logs.
8. Open **Review** or **Final Video** to inspect outputs.

CLI equivalent for an existing project:

```bash
uv run python run_pipeline.py ./projects/my-song --skip-tests
```

### Video Pipeline Modes

Three rendering pipelines control how the video is generated:

```bash
# Classic — storyboard/start-frame driven
uv run python run_pipeline.py ./projects/my-song \
  --video-pipeline ltx_i2v --skip-tests

# MSR — actor/location reference-sheet driven
uv run python run_pipeline.py ./projects/my-song \
  --video-pipeline ltx_msr --skip-tests

# Ingredients — per-scene reference sheets with audio latent
uv run python run_pipeline.py ./projects/my-song \
  --video-pipeline ltx_ingredients --skip-tests
```

### Render Modes

Each pipeline renders prompts using one of three modes:

```bash
# single_prompt — patches #PROMPT (default, simplest)
uv run python run_pipeline.py ./projects/my-song \
  --render-mode single_prompt --skip-tests

# relay — patches #PROMPT_RELAY for multi-scene continuity
uv run python run_pipeline.py ./projects/my-song \
  --render-mode relay \
  --relay-workflow ./workflows/video_ltxv_relay_v1.json --skip-tests

# auto — picks per-scene from the render plan hints
uv run python run_pipeline.py ./projects/my-song \
  --render-mode auto \
  --single-prompt-workflow ./workflows/video_ltxv_i2v_v1.json \
  --relay-workflow ./workflows/video_ltxv_relay_v1.json --skip-tests
```

### Movie Pipeline Modes

Four movie rendering workflows control shot generation:

```bash
# MSR — reference-sheet driven (default)
uv run python movie_pipeline.py ./projects/my-movie

# I2V/Edit — classic edit-workflow driven
uv run python movie_pipeline.py ./projects/my-movie \
  --movie-video-workflow i2v-edit

# Startframe Director — multi-stage (Krea2/Ideogram -> mask -> repair -> detail -> validate -> LTX I2V)
uv run python movie_pipeline.py ./projects/my-movie \
  --movie-video-workflow startframe-director \
  --startframe-director-backend krea2

# Ingredients — per-shot target prompts through an ingredients workflow
uv run python movie_pipeline.py ./projects/my-movie \
  --movie-video-workflow ingredients
```

### Rendering Tweaks

Control frame rolling, LoRA handling, and seeding:

```bash
# Low-VRAM rolling profile (6 preroll, 0 tail)
uv run python run_pipeline.py ./projects/my-song \
  --rolling-frame-profile safe --skip-tests

# LoRA split (halve base strength, add second anchor at full strength)
uv run python run_pipeline.py ./projects/my-song \
  --lora-split-enabled --skip-tests

# Random seed per scene instead of deterministic scene-number seeds
uv run python run_pipeline.py ./projects/my-song \
  --randomize-seed --skip-tests
```

## Full-Auto

Full-Auto creates a project from a short idea and song style, renders ACE-Step audio through ComfyUI, writes `config.json`, and can immediately run the video pipeline.

Three pipeline modes are available:

- **Classic** (`ltx_i2v`): storyboard/start-frame driven rendering.
- **MSR** (`ltx_msr`): reference-sheet driven rendering with actor/location identity.
- **Ingredients** (`ltx_ingredients`): per-scene ingredients sheets with audio latent injection.

Studio asks for:

- project name
- idea
- song style
- desired video duration
- width and height, default `1280 x 704`
- FPS, default `24`, allowed `16`, `24`, `50`
- pipeline mode, **MSR** or **Classic**

CLI example:

```bash
uv run python full_auto.py \
  --idea "A cyberpunk chase through a futuristic city" \
  --style "dark synthwave with cinematic drums" \
  --project-name neon-wolves \
  --duration-seconds 120 \
  --width 1280 \
  --height 704 \
  --fps 24 \
  --run-video-pipeline \
  --video-pipeline ltx_msr \
  --skip-tests
```

Full-Auto with ingredients mode:

```bash
uv run python full_auto.py \
  --idea "A cyberpunk chase through a futuristic city" \
  --style "dark synthwave with cinematic drums" \
  --project-name neon-wolves \
  --run-video-pipeline \
  --video-pipeline ltx_ingredients \
  --skip-tests
```

### LTX 2.3 Ingredients geometry

The Ingredients workflows use two-stage sampling. For the model's recommended
bucket, FeverSlop passes the final target size `1536 x 896`; the workflow
samples stage 1 at `768 x 448` and reaches the target through the LTX 2.3 x2
spatial latent upscaler. Ingredients clips should use at least `121` frames at
`24` FPS. Reference sheets are composed at a high-resolution `12:7` canvas and
then downscaled to the stage-1 conditioning size.

See the official [Lightricks LTX-2.3 Ingredients model card](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients)
and the [reference Space implementation](https://huggingface.co/spaces/ltx-community/ltx-2.3-ingredients-distilled/tree/main).

## Requirements

Most useful runs require:

- FFmpeg in `PATH`.
- ComfyUI running and reachable from `app_config.json`.
- Required ComfyUI custom nodes/models for the selected workflows.
- `workflows/audio_song_v2.json` with ACE-Step nodes for Full-Auto audio generation.
- An OpenAI-compatible LLM endpoint configured in `app_config.json`.

## Documentation

- [Setup](docs/setup.md): prerequisites, dependencies, ComfyUI/ACE-Step, local config.
- [Running](docs/running.md): backend/frontend startup, ports, production build notes.
- [Pipelines](docs/pipelines.md): standard, Full-Auto, MSR/classic, progress and logs.
- [Projects](docs/projects.md): folder structure, `config.json`, Project Settings, artifacts.
- [Examples](docs/examples.md): standard and Full-Auto workflows, API calls, troubleshooting.
- [Project workflow deep reference](docs/project_workflow.md): detailed legacy CLI reference.
- [ComfyUI model resolution](docs/comfyui_model_resolution.md): portable workflow model matching.

## Verification Commands

Backend/tests:

```bash
uv run ruff check .
uv run python -m unittest discover -s tests
```

Frontend:

```bash
cd studio/frontend
bun test src
bun run build
bun run test:e2e
```
