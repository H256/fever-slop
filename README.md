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
7. Open **Pipeline**, start a job, and monitor progress/logs.
8. Open **Review** or **Final Video** to inspect outputs.

CLI equivalent for an existing project:

```bash
uv run python run_pipeline.py ./projects/my-song --skip-tests
```

## Full-Auto

Full-Auto creates a project from a short idea and song style, renders ACE-Step audio through ComfyUI, writes `config.json`, and can immediately run the video pipeline.

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

## Requirements

Most useful runs require:

- FFmpeg in `PATH`.
- ComfyUI running and reachable from `app_config.json`.
- Required ComfyUI custom nodes/models for the selected workflows.
- `workflows/audio_song.json` with ACE-Step nodes for Full-Auto audio generation.
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
