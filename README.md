![<img height="480" src="./feverslop_logo.png" width="480"/>](./feverslop_logo.png)

# FeverSlop Music Video Pipeline

FeverSlop turns a song into a beat- and vocal-aware music-video render plan, then renders storyboard frames, LTX image-to-video clips, and a final video muxed with the original full-song audio.

The current default LTX path is the non-relay `single_prompt` mode. PromptRelay remains available for workflows with a correctly wired `#PROMPT_RELAY` node, and `auto` mode can choose between relay and single-prompt rendering per scene when both workflows are configured.

Full project setup, config keys, all runner parameters, steering examples, LoRA behavior, and debugging notes live in:

```text
docs/project_workflow.md
```

## Requirements

- Python 3.12
- `uv`
- FFmpeg in `PATH`
- ComfyUI with the required API workflows and models
- an OpenAI-compatible LLM endpoint
- Demucs and Whisper dependencies from the Python environment
- for Full-Auto: an ACE-Step 1.5 ComfyUI workflow at `workflows/audio_song.json`

Install dependencies:

```powershell
uv sync
```

## Quick Start

For an existing project that already has `config.json` and an input audio file:

```powershell
uv run python run_pipeline.py ./projects/my_song
```

Typical project layout:

```text
projects/my_song/
|-- config.json
|-- input/
|   `-- my_song.mp3
`-- output/
   |-- stems/
   |-- timeline/
   |-- prompts/
   `-- render/
      |-- render_plan_my_song.json
      |-- storyboard/
      `-- ltx_single_prompt/
```

The final muxed video is normally written under:

```text
projects/my_song/output/render/ltx_single_prompt/<project_name>.mp4
```

## Full-Auto

`full_auto.py` creates a complete FeverSlop project from an idea and rough musical style. It generates a structured ACE-Step song brief, lyrics, an MP3 via `workflows/audio_song.json`, `config.json`, and optionally starts the normal video pipeline.

```powershell
uv run python full_auto.py `
  --idea "friendship and joy in a bright city" `
  --style "upbeat contemporary pop, warm, bright, catchy, male vocal" `
  --project-name joy_demo `
  --duration-seconds 120 `
  --width 1280 `
  --height 704 `
  --language en `
  --seed 42
```

Generated files:

```text
projects/joy_demo/
|-- config.json
|-- full_auto_song_spec.json
|-- lyrics.txt
`-- input/
    `-- joy_demo.mp3
```

To immediately run a smoke video render after the song/project is created:

```powershell
uv run python full_auto.py `
  --idea "friendship and joy in a bright city" `
  --style "upbeat contemporary pop, warm, bright, catchy, male vocal" `
  --project-name joy_demo `
  --duration-seconds 120 `
  --width 1280 `
  --height 704 `
  --language en `
  --seed 42 `
  --run-video-pipeline `
  --skip-tests `
  --smoke-only `
  --smoke-scene 1 `
  --rolling-frame-profile safe
```

Full-Auto accepts the runner override surface from `run_pipeline.py`, including workflow paths, `--render-mode`, single-prompt node settings, LoRA overrides, smoke flags, skip flags, concat flags, and `--rolling-frame-profile`.

Example with storyboard and video LoRA strength overrides:

```powershell
uv run python full_auto.py `
  --idea "friendship and joy in a bright city" `
  --style "upbeat contemporary pop, warm, bright, catchy, male vocal" `
  --project-name joy_demo `
  --run-video-pipeline `
  --storyboard-lora-strength 0.4 `
  --video-lora-1-strength-model 0.7 `
  --video-lora-1-strength-clip 0.6 `
  --lora-split-enabled `
  --rolling-frame-profile safe `
  --skip-tests
```

`--width` and `--height` write `video.width` and `video.height` into the generated project config, so prompt generation, storyboard rendering, and video rendering use that resolution.

## Full-Auto ACE-Step Workflow Contract

`workflows/audio_song.json` must contain these API workflow nodes by `_meta.title`:

| Node title | Class | Patched inputs |
| --- | --- | --- |
| `ACE_STEP` | `TextEncodeAceStepAudio1.5` | `tags`, `lyrics`, `bpm`, `duration`, `language`, `keyscale`, `timesignature`, `seed` |
| `KSampler` | `KSampler` | `seed` |
| `Empty Ace Step 1.5 Latent Audio` | `EmptyAceStep1.5LatentAudio` | `seconds` |
| `SAVE` | `SaveAudioMP3` | `filename_prefix` |

`ACE_STEP.duration` and `Empty Ace Step 1.5 Latent Audio.seconds` are always patched together from `--duration-seconds`. `ACE_STEP.seed` and `KSampler.seed` are always patched together from `--seed`. Technical sampling values such as `steps`, `cfg`, `cfg_scale`, `temperature`, `top_p`, `top_k`, `min_p`, `sampler_name`, `scheduler`, and `denoise` stay owned by the workflow in v1.

## Configuration

Global infrastructure config lives at:

```text
app_config.json
```

Example:

```json
{
  "llm": {
    "base_url": "http://localhost:8080/v1",
    "model": "default",
    "temperature": 0.7,
    "max_tokens": 4096
  },
  "comfyui": {
    "base_url": "http://127.0.0.1:8188",
    "prompt_timeout_seconds": 1800,
    "model_overrides": []
  }
}
```

Project config lives inside each project:

```text
projects/my_song/config.json
```

Minimal project config:

```json
{
  "project_name": "my_song",
  "input_audio": "input/my_song.mp3",
  "lyrics": "",
  "video": {
    "fps": 24,
    "width": 1280,
    "height": 704
  }
}
```

Important project fields:

- `input_audio`: audio path relative to `config.json`, or absolute.
- `lyrics`: optional reference lyrics used to correct Whisper text while preserving detected timing.
- `story_idea`, `style`, `subject`, `locations`: hard visual overrides for prompt generation.
- `steering.*`: soft instructions injected into specific LLM stages.
- `prompt_guidance.*`: reusable prompt vocabulary for scene, image, and video prompt generation.
- `loras` and `lora_split_enabled`: project-level LTX LoRA defaults used by `render_ltx.py` and `run_pipeline.py`.

## Runner Overview

`run_pipeline.py` performs:

1. optional unit tests
2. main render-plan generation
3. optional relay prompt compaction
4. prompt anchor fixing
5. storyboard rendering
6. storyboard review page generation
7. LTX rendering
8. video-only concat
9. original full-audio mux

For `--video-pipeline ltx_msr`, the runner skips storyboard/startframe generation, renders actor and location MSR references, writes a `_refs.json` render plan, renders MSR clips, then runs the same final concat and original-audio mux.

Common runner commands:

```powershell
uv run python run_pipeline.py ./projects/my_song --smoke-only --smoke-scene 3
```

```powershell
uv run python run_pipeline.py ./projects/my_song --no-skip-existing --rolling-frame-profile safe
```

```powershell
uv run python run_pipeline.py --project-config ./projects/my_song/config.json --skip-tests
```

The runner's command-line values override project config values where supported. For LTX rendering, `render_ltx.py` can read scene-duration and LoRA defaults from `config.json`, while explicit CLI values still win.

Run an MSR project end to end:

```powershell
uv run python run_pipeline.py ./projects/dwarfventure-msr `
  --video-pipeline ltx_msr `
  --skip-tests
```

Reuse existing MSR references after editing only the video workflow:

```powershell
uv run python run_pipeline.py ./projects/dwarfventure-msr `
  --video-pipeline ltx_msr `
  --skip-msr-reference-render `
  --no-skip-existing `
  --skip-tests
```

Render a fresh variant for one scene with a new seed:

```powershell
uv run python run_pipeline.py ./projects/dwarfventure-msr `
  --video-pipeline ltx_msr `
  --skip-msr-reference-render `
  --randomize-seed `
  --no-skip-existing `
  --smoke-only `
  --smoke-scene 4 `
  --skip-tests
```

## Package Layout

Root scripts are public CLIs or compatibility facades. New implementation code lives under `src/feverslop`.

```text
src/feverslop/
|-- application/    use cases and pipeline services without concrete adapters
|-- composition/    wiring for configs, use cases, and adapters
|-- domain/         render-plan, LTX, postprocessing, and Full-Auto domain types
|-- ports/          protocols and request types
|-- adapters/       ComfyUI, local artifacts, LLM, FFmpeg, and runner adapters
|-- pipeline/       render-plan and timeline builders
|-- prompting/      prompt generation and prompt repair
`-- tools/          importable tool implementations
```

Important composition roots:

```text
feverslop.composition.generate_render_plan
feverslop.composition.render_storyboard
feverslop.composition.render_video
feverslop.composition.full_auto
```

The current production LTX render adapter is:

```text
feverslop.adapters.comfyui_video_backend
```

The root files `ltx_video_renderer.py`, `storyboard_renderer.py`, and `workflow_patcher.py` are compatibility facades for older imports. See:

```text
docs/architecture_compatibility.md
```

## Useful Manual Commands

Generate the main render plan only:

```powershell
uv run python main.py `
  --project ./projects/my_song/config.json `
  --app-config ./app_config.json `
  --concept-batch-size 10
```

Render storyboard frames:

```powershell
uv run python render_storyboard.py `
  --app-config ./app_config.json `
  --render-plan ./projects/my_song/output/render/render_plan_my_song.json `
  --workflow ./workflows/image_t2i_startframe_v1.json `
  --output-dir ./projects/my_song/output/render/storyboard
```

Render one LTX smoke scene:

```powershell
uv run python render_ltx.py `
  --project-config ./projects/my_song/config.json `
  --app-config ./app_config.json `
  --render-plan ./projects/my_song/output/render/render_plan_my_song.json `
  --workflow ./workflows/video_ltxv_i2v_v1.json `
  --render-mode single_prompt `
  --audio ./projects/my_song/input/my_song.mp3 `
  --storyboard-dir ./projects/my_song/output/render/storyboard `
  --output-dir ./projects/my_song/output/render/ltx_single_prompt_smoke `
  --scenes 1 `
  --no-skip-existing `
  --debug-workflows-dir ./projects/my_song/output/render/ltx_single_prompt_debug
```

Create a project asset archive:

```powershell
uv run python -m tools.project_asset_archive --project ./projects/my_song/config.json
```

## Debugging

- If LLM JSON is truncated, increase `llm.max_tokens` or use `--concept-batch-size 5` or `10`.
- If LTX ignores the start frame, make sure storyboard and LTX use the same final render plan, run the prompt anchor fixer, and inspect the saved debug workflow JSON.
- If vocals are detected incorrectly, inspect `output/timeline/timeline_<song>.json` and tune `vocal_detection` in `config.json`.
- If clips are too short or too long, regenerate the render plan after adjusting `scene_generation.min_duration` and `max_duration`; do not repair only at LTX time.
- If ComfyUI rejects a workflow, check the exact missing `_meta.title` anchor and inspect the debug workflow.

## Verification

Run the full test suite:

```powershell
uv run python -m unittest discover -s tests
```

Check tracked changes:

```powershell
git status --short
```
