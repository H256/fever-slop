# FeverSlop

Public repository: https://github.com/H256/fever-slop

FeverSlop is a local, CLI-first music-video generation pipeline. It turns an
audio track, lyrics, and visual direction into project artifacts, render plans,
ComfyUI image/video jobs, reviewable clips, and a final muxed video.

> [!WARNING]
> FeverSlop is actively developed and is currently intended for experimenters
> and tinkerers. It is not a finished, turnkey product, and the workflows,
> model requirements, configuration, and CLI may change. Expect to inspect
> artifacts, adjust prompts and workflows, and solve local ComfyUI/model setup
> issues yourself.

The goal is not to provide a polished end-to-end production system. FeverSlop
provides building blocks that make prompt generation, render-plan creation,
ComfyUI workflow preparation, and repeatable local rendering easier to inspect,
combine, and automate. Use the parts that fit your setup and adapt the rest.

The primary documented operator surface is the command line:
`run_pipeline.py`, `full_auto.py`, and `movie_pipeline.py`. The former Studio
application is deprecated and is not a supported user interface; a few older
internal components may still be reused by the active code. Projects are
ordinary directories containing JSON artifacts, media, and configuration, so
they can be inspected and automated with standard shell tools.

Core Python packages live under `src/feverslop`. Composition code such as
`feverslop.composition.generate_render_plan` is kept separate from adapters
such as `feverslop.adapters.comfyui_video_backend`.

The DSPy prompt boundary, bundled guide naming, typed inputs, fallback rules,
and concurrency model are documented in [documentation/prompt-architecture.md](documentation/prompt-architecture.md).

The canonical global asset library, project snapshots, guided generation, and
prop interactions are documented in [documentation/global-assets.md](documentation/global-assets.md).

## Quick Start

Install Python dependencies from the repository root:

```bash
uv sync
```

The default `uv` configuration selects the PyTorch CUDA 13.0 index for the
Linux/Windows GPU workflow. That index does not publish macOS wheels. On
macOS, install the CPU/Metal-compatible packages from the normal Python
indexes by ignoring the project-specific source overrides:

```bash
uv sync --no-sources
```

Do not use the CUDA-specific ComfyUI/PyTorch workflow on macOS unless you have
provided a separate, compatible local setup.

#### Windows PowerShell: `litellm` file-lock error

If `uv sync` fails on Windows with `Failed to install: litellm` and
`os error 32` while renaming a file under `.venv\Lib\site-packages\litellm`,
a running process or Windows scanning service is holding the file temporarily.
Close ComfyUI, FeverSlop, and IDE terminals that use this environment, then
recreate the disposable virtual environment:

```powershell
deactivate 2>$null
Remove-Item -LiteralPath .venv -Recurse -Force
uv cache clean litellm
uv sync --link-mode=copy --refresh-package litellm --reinstall-package litellm
```

If `.venv` is still locked, close the IDE and all repository terminals, restart
PowerShell (or Windows), and run the commands again. If the error keeps
recurring after a restart, identify the locking process with Resource Monitor
or Process Explorer before considering a narrowly scoped Windows Defender
exclusion for the local `.venv` directory.

## Basic Workflows

### Standard Music Video Project

1. Create a project directory under `projects/` and add its `config.json`.
2. Set `input_audio` and choose `video_pipeline`:
   - `ltx_msr`: MSR/reference-guided mode.
   - `ltx_i2v`: classic storyboard/start-frame mode.
   - `ltx_ingredients`: per-scene ingredients sheets with audio latent injection.
3. Run the pipeline and inspect the generated artifacts:

```bash
uv run python run_pipeline.py ./projects/my-song --skip-tests
```

#### Reusing audio analysis artifacts

The main pipeline can reuse individual audio-analysis artifacts instead of
running every expensive sub-step again. The corresponding files must already
exist under `output/`:

```bash
uv run python run_pipeline.py ./projects/my-song \
  --skip-stem-separation \
  --skip-whisper \
  --skip-beat-analysis
```

These flags can be combined or used independently:

- `--skip-stem-separation` reuses the four existing Demucs stem files.
- `--skip-whisper` reuses `output/timeline/timeline_<song>.json`.
- `--skip-beat-analysis` reuses `output/timeline/beat_data_<song>.json`.

If a requested artifact is missing, the pipeline stops with the exact path it
needs instead of silently rerunning that step. For explicit stage selection,
`prepare_workflows` and `render_scenes` are the backend-agnostic names; the
legacy spellings `ltx_prepare_workflows` and `ltx_render_scenes` remain
supported.

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

For MiniMax H3 R2V, the default pipeline generates MSR references and reference
sheets before stage 8.5. DSPy can then analyze the actual actor/location images
before the final render plan is written. Stage 8.5 can also be rerun without
repeating the earlier LLM stages, but it requires the reference-enriched plan
from `msr_reference_sheets`; run `render_plan` afterward to copy the new prompts
into `base.json`:

For the sequence-based actor/location sheet path, see the complete
[Sequence-to-Sheet Reference Pipeline tutorial](documentation/sequence-reference-pipeline.md).
It documents the opt-in `--reference-generation sequence_sheet` mode for both
music-video and Movie projects, including direct reference rebuilding and
reviewable intermediate artifacts.

To generate the MSR sheets with a custom image workflow and stop after preparing
the final MiniMax R2V render plan, select the stages explicitly:

```bash
uv run python run_pipeline.py ./projects/my-song \
  --video-pipeline minimax-h3-r2v \
  --reference-hero-workflow workflows/my_custom_reference_workflow.json \
  --stage main_pipeline \
  --stage anchor_fix \
  --stage msr_references \
  --stage msr_reference_sheets \
  --stage h3_prompts \
  --stage render_plan \
  --skip-tests
```

Add `--reference-edit-workflow workflows/my_custom_edit_workflow.json` when the
reference workflow also uses a separate edit pass. This command renders the MSR
reference images, but does not render any video. It stops after writing the
reference-aware MiniMax prompts and final render plan. MiniMax does not currently
support the `ltx_prepare_workflows` stage, so per-scene `workflow.json` and
`manifest.json` files are created only when the video render stage runs.

```bash
uv run python run_pipeline.py ./projects/my-song \
  --video-pipeline minimax-h3-r2v \
  --stage h3_prompts --skip-tests

uv run python run_pipeline.py ./projects/my-song \
  --video-pipeline minimax-h3-r2v \
  --stage render_plan --skip-tests
```

#### DSPy prompt generation for MiniMax H3

MiniMax H3 prompt generation uses the DSPy generator automatically for the
supported MiniMax H3 modes:

```bash
uv run python run_pipeline.py ./projects/my-song \
  --video-pipeline minimax-h3-t2v --skip-tests
```

The standalone tool prints the generated prompt to stdout. It does not write
project artifacts; an output option may be added in a future version.
DSPy calls share the same process-local LLM concurrency budget as direct
OpenAI-compatible calls. Configure it with `llm.max_concurrent_requests` in
`app_config.json`; the default is `1` and applies only inside the current
FeverSlop process, not across multiple processes or other clients.

```bash
# T2V: text only; no reference is required
uv run python tools/generate_prompt.py \
  --model-type minimax-h3-t2v \
  --description "A singer walks through a neon-lit city at night."

# I2V: exactly one picture with the first_frame role is required
uv run python tools/generate_prompt.py \
  --model-type minimax-h3-i2v \
  --description "The singer looks toward the camera and starts walking." \
  --reference '{"kind":"picture","source":"start.png","role":"first_frame"}'

# R2V: supply the reference assets that should guide the shot
uv run python tools/generate_prompt.py \
  --model-type minimax-h3-r2v \
  --description "The singer performs in the same neon-lit city." \
  --reference '{"kind":"picture","source":"singer.png","role":"subject"}' \
  --reference '{"kind":"picture","source":"city.png","role":"environment"}'
```

Reference roles describe how each asset is used: `subject` identifies a
person or object, `style` transfers visual treatment, `environment` anchors a
place, and `motion` or `camera` guides movement or framing. Mode-specific
requirements are strict: I2V needs exactly one picture with `first_frame`,
FL2V needs one each with `first_frame` and `last_frame`, and L2V needs exactly
one picture with `last_frame`. T2V has no required reference role. R2V uses
the reference-aware guide and accepts roles such as `subject`, `environment`,
`style`, `motion`, and `camera` for the assets supplied.

The bundled `minimax-h3-base.md` guide defines prompt construction for T2V,
I2V, FL2V, and L2V. The `minimax-h3-references.md` guide is reserved for R2V
and explains how multiple reference assets and their roles are incorporated.

MSR and Ingredients runs materialize each selected scene before rendering. To
inspect the exact ComfyUI workflows without queueing a render, run the prepare
stage explicitly; use the same `--scenes` filter for the later render stage:

```bash
uv run python run_pipeline.py ./projects/my-song \
  --video-pipeline ltx_ingredients \
  --stage ltx_prepare_workflows \
  --scenes 1,3,5

uv run python run_pipeline.py ./projects/my-song \
  --video-pipeline ltx_ingredients \
  --stage ltx_render_scenes \
  --scenes 1,3,5
```

Prepared assets live under `output/render/scenes/scene_####/`; rendering
verifies their manifest hashes before sending the stored `workflow.json` to
ComfyUI.

### Render Modes

Each pipeline renders prompts using one of three modes:

```bash
# single_prompt — patches #PROMPT (default, simplest)
uv run python run_pipeline.py ./projects/my-song \
  --render-mode single_prompt --skip-tests

# relay — patches #PROMPT_RELAY for multi-scene continuity
uv run python run_pipeline.py ./projects/my-song \
  --render-mode relay \
  --relay-workflow ./workflows/your_prompt_relay_workflow.json --skip-tests

# auto — picks per-scene from the render plan hints
uv run python run_pipeline.py ./projects/my-song \
  --render-mode auto \
  --single-prompt-workflow ./workflows/video_ltxv_i2v_v2.json \
  --relay-workflow ./workflows/your_prompt_relay_workflow.json --skip-tests
```

`your_prompt_relay_workflow.json` is an external/custom API workflow
placeholder; no relay template is bundled in this repository.

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

For Movie MSR and Ingredients projects, `--write-debug-workflows` is the
deprecated compatibility alias for preparing the canonical scene workflows
without queueing them. `--scenes 1,3,5` limits both preparation and rendering.

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

### Postprocessing: FaceFix

After rendering, faces can drift across frames. The FaceFix stage applies a
lightweight video-to-video refinement pass using the LTXV LoopingSampler with
periodic face keyframe conditioning:

```bash
# Run FaceFix on all rendered scenes
uv run python run_pipeline.py ./projects/my-song \
  --stage facefix

# Run FaceFix with face reference images
uv run python run_pipeline.py ./projects/my-song \
  --stage facefix \
  --facefix-reference-images ./projects/my-song/face_ref_01.jpg \
  --facefix-reference-images ./projects/my-song/face_ref_02.jpg

# Override keyframe indices and guiding strength
uv run python run_pipeline.py ./projects/my-song \
  --stage facefix \
  --facefix-keyframe-indices "0,8,24,40" \
  --facefix-guiding-strength 0.3
```

The stage reads `scene_NNNN.mp4` files from the render output directory, applies
temporal face conditioning at configurable keyframe intervals, and outputs
`scene_NNNN_facefix.mp4` files alongside the originals. All face reference images
are passed to the LTXV LoopingSampler, which handles per-face matching internally:
it detects faces in each keyframe and matches them to the closest reference. This
means you can provide references for all characters and the workflow applies the
correct one to each detected face.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--facefix-reference-images` | none | Path(s) to face reference images from the movie bible |
| `--facefix-keyframe-indices` | `0,16,32,48` | Frames where face conditioning is applied |
| `--facefix-guiding-strength` | `0.2` | Strength of face guidance during sampling |
| `--facefix-cond-image-strength` | `0.5` | Strength of reference image conditioning |
| `--facefix-temporal-tile-size` | `56` | Frames processed per temporal tile |
| `--facefix-temporal-overlap` | `24` | Overlap between temporal tiles |

Without reference images, the stage runs video-only conditioning, which still
stabilizes faces but without character-specific guidance.

## Full-Auto

Full-Auto creates a project from a short idea and song style, renders ACE-Step audio through ComfyUI, writes `config.json`, and can immediately run the video pipeline.

The following pipeline modes are available:

- **Classic** (`ltx_i2v`): storyboard/start-frame driven rendering.
- **MSR** (`ltx_msr`): reference-sheet driven rendering with actor/location identity.
- **Ingredients** (`ltx_ingredients`): per-scene ingredients sheets with audio latent injection.
- **MiniMax H3 R2V** (`minimax-h3-r2v`): reference-to-video rendering with actor/location
  references and optional audio-stem references.
- **MiniMax H3 I2V** (`minimax-h3-i2v`): image-to-video rendering from the generated
  scene start frames.
- **MiniMax H3 T2V** (`minimax-h3-t2v`): text-to-video rendering without an image reference.

Full-Auto asks for:

- project name
- idea
- song style
- desired video duration
- width and height, default `1280 x 704`
- FPS, default `24`, allowed `16`, `24`, `50`
- pipeline mode, including **Classic**, **MSR**, **Ingredients**, and MiniMax H3
  **R2V**, **I2V**, or **T2V**

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

Full-Auto with MiniMax H3:

```bash
uv run python full_auto.py \
  --idea "A cyberpunk chase through a futuristic city" \
  --style "dark synthwave with cinematic drums" \
  --project-name neon-wolves \
  --run-video-pipeline \
  --video-pipeline minimax-h3-r2v \
  --skip-tests
```

Use `minimax-h3-i2v` or `minimax-h3-t2v` instead of `minimax-h3-r2v` to select
the corresponding MiniMax H3 input mode. R2V can additionally receive selected
audio stems through `--minimax-audio-ref-stems`, for example
`vocals,drums,bass`.

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

- [Documentation index](documentation/README.md): user-facing documentation overview.
- [App configuration reference](documentation/app_config.md): complete `app_config.json` field reference and defaults.
- [LLM performance](documentation/llm-performance.md): Thinking trade-offs, token diagnostics, and concurrency guidance.
- [LLM benchmark](documentation/llm-benchmark.md): compare server-side Thinking configurations.
- [Setup](documentation/setup.md): prerequisites, dependencies, ComfyUI/ACE-Step, local config.
- [Running](documentation/running.md): CLI startup and pipeline operation.
- [Pipelines](documentation/pipelines.md): standard, Full-Auto, MSR/classic, progress and logs.
- [Projects](documentation/projects.md): folder structure, `config.json`, Project Settings, artifacts.
- [Examples](documentation/examples.md): standard and Full-Auto workflows and troubleshooting.
- [Project workflow deep reference](documentation/project_workflow.md): detailed legacy CLI reference.
- [ComfyUI model resolution](documentation/comfyui_model_resolution.md): portable workflow model matching.
- [Workflow model requirements](documentation/workflow-models.md): model filenames, loader roles, and workflow coverage.
- [Contributing](CONTRIBUTING.md): development setup and contribution guidelines.
- [Code of Conduct](CODE_OF_CONDUCT.md): community participation standards.
- [Security policy](SECURITY.md): private vulnerability reporting guidance.
- [Third-party notices](THIRD_PARTY_NOTICES.md): external runtimes, models, and assets.
- [MIT License](LICENSE): license for the FeverSlop source code.

## Verification Commands

Python tests and linting:

```bash
uv run ruff check .
uv run python -m unittest discover -s tests
```

Dependency security audit:

```bash
uvx pip-audit .
```

To run the same check automatically before committing, install the checked-in
hook configuration once with `uvx pre-commit install`.
