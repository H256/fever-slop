# Project Workflow Guide

This guide documents how to create a new FeverSlop music-video project, which commands to run, what each config setting does, and which optional settings steer the visual result.

The recommended path is:

1. Create a project folder with `config.json` and an input audio file.
2. Run `run_pipeline.py` with the project root path.
3. Inspect the storyboard and one LTX smoke scene.
4. Run the full LTX render.
5. Use the generated `<project_name>.mp4`, which is video-only concatenation muxed with the original full song audio.

## Repository Setup

Install dependencies from the repository root:

```powershell
uv sync
```

Required external tools:

- `uv`
- FFmpeg in `PATH`
- ComfyUI running with the required workflows and models
- an OpenAI-compatible LLM endpoint
- Demucs and Whisper dependencies installed through the Python environment

Global infrastructure config lives at:

```text
app_config.json
```

Project config lives inside each project:

```text
projects/my_song/config.json
```

## New Project Structure

Create this layout:

```text
projects/my_song/
|-- config.json
|-- input/
|   `-- my_song.mp3
`-- output/
```

`output/` is created automatically if it does not exist.

Start from the example config:

```powershell
New-Item -ItemType Directory -Force ./projects/my_song/input
Copy-Item ./config.example.json ./projects/my_song/config.json
```

Then edit:

```json
{
  "project_name": "my_song",
  "input_audio": "input/my_song.mp3",
  "lyrics": ""
}
```

Paths inside `config.json` are resolved relative to the config file.

## One-Command Runner

For normal operation on an existing project, preview the safe work first and
then resume exactly the required phases:

```powershell
uv run python main.py run ./projects/my_song --dry-run
uv run python main.py run ./projects/my_song --resume
```

The immutable plan is computed from `base.json`, per-scene canonical
fingerprints, H3 checkpoints, prepared workflow manifests, clips, and final
outputs. Actions are `RUN`, `REUSE`, `BLOCKED`, and `NOT_SELECTED`; every RUN or
BLOCKED row includes its reason. A prompt-only change remains scene-local, a
reference/timing/resolution change expands through the required dependencies,
and valid partial H3/render output is reused. BLOCKED legacy edits must be
resolved with the displayed `plan-migrate`/`plan validate` command before any
stage can start.

If a real run fails, its summary records the last completed stage and prints
the exact `main.py run ... --resume` recovery command. The plan is recalculated
on resume, so newly completed artifacts become REUSE.

### Human correction checkpoint

`output/render/plans/base.json` is the sole editable render-plan artifact. Use
`uv run python main.py plan show ./projects/my_song --scene N` to compare the
generator-owned value, the optional human-owned override, and the effective
value. Put an intentional correction only in
`canonical.roles.<role>.override`; do not replace `generated` and do not add a
persisted `effective` field.

Scene-local `h3_prompt.json` files make judged H3 output inspectable as each
scene completes, but remain generated checkpoints. The aggregate H3 file,
`compact.json`, `anchored.json`, `references.json`, `ingredients.json`, scene
manifests, and prepared workflows are also derived artifacts. After editing
`base.json`, use the normal dry-run/resume pair with `--scenes N`; it selects
reference enrichment, preparation, rendering, and assembly according to the
scene-local fingerprints. See the complete
[human prompt correction workflows](running.md#human-prompt-correction-workflows).

### Advanced and compatibility runner

Python:

```powershell
uv run python run_pipeline.py ./projects/my_song
```

The first positional argument is the project root containing `config.json`. Passing the config file directly also works:

```powershell
uv run python run_pipeline.py ./projects/my_song/config.json
```

The legacy/advanced runner performs:

1. optional unit tests
2. `main.py`
3. optional relay prompt compaction
4. prompt anchor fixing
5. storyboard rendering
6. storyboard review page generation
7. LTX rendering
8. video-only concat
9. original full-audio mux

Optional SeedVR2 upscale can be enabled after rendering and FaceFix:

```powershell
uv run python run_pipeline.py ./projects/my_song --upscale --skip-tests
```

For repeatable project settings, put the `upscale` object in the project's
`config.json`. SeedVR2 currently has no global application-level configuration
surface; the project object is the single persistent configuration layer. Use exactly one of
`target_width`, `target_height`, or both, or neither for the default 2x target.
For a one-off override, use `--upscale-resolution WIDTHxHEIGHT`.

`--stage upscale` selects only the upscale stage. It does not enable SeedVR2
when the project configuration has `upscale.enabled` set to `false` or has no
upscale configuration. Use `--upscale` as a one-off enable override in that
case. If the project already enables upscaling, `--stage upscale` alone is
enough; `--upscale` is then redundant. Use `--no-skip-existing` to force a
re-render of existing upscale artifacts.

The upscale stage logs every pass and its dimensions in each scene's
`upscale_manifest.json`. It creates `upscale_final.mp4` without replacing the
original scene artifact. Existing artifacts are safe to reuse, so a failed
run can be resumed with the same command.

Configuration precedence is:

1. built-in defaults
2. project `config.json`
3. explicit command-line arguments

The final file is named after sanitized `project_name`:

```text
projects/my_song/output/render/ltx_single_prompt/my_song.mp4
```

## Full-Auto Runner

`full_auto.py` creates a project from an idea and rough style, then optionally calls the normal one-command runner.

```powershell
uv run python full_auto.py `
  --idea "friendship and joy in a bright city" `
  --style "upbeat contemporary pop, warm, catchy, male vocal" `
  --project-name joy_demo `
  --duration-seconds 120 `
  --width 1280 `
  --height 704 `
  --fps 24 `
  --language en `
  --run-video-pipeline `
  --video-pipeline ltx_msr `
  --skip-tests
```

Outputs:

```text
projects/joy_demo/
|-- config.json
|-- full_auto_song_spec.json
|-- lyrics.txt
`-- input/
    `-- joy_demo.mp3
```

ACE-STEP workflow contract for `workflows/audio/audio-model/audio_song_v2.json`:

| Node title | Class | Patched inputs |
| --- | --- | --- |
| `ACE_STEP` | `TextEncodeAceStepAudio1.5` | `tags`, `lyrics`, `bpm`, `duration`, `language`, `keyscale`, `timesignature`, `seed` |
| `KSampler` | `KSampler` | `seed` |
| `Empty Ace Step 1.5 Latent Audio` | `EmptyAceStep1.5LatentAudio` | `seconds` |
| `SAVE` | `SaveAudioMP3` | `filename_prefix` |

`duration` and `seconds` are always patched together from `--duration-seconds`. `ACE_STEP.seed` and `KSampler.seed` are always patched together from `--seed`. Sampling parameters such as `steps`, `cfg`, `cfg_scale`, `temperature`, `top_p`, `top_k`, `min_p`, `sampler_name`, `scheduler`, and `denoise` stay owned by the workflow in v1.

`--width`, `--height`, and `--fps` are optional Full-Auto project settings. They write `video.width`, `video.height`, and `video.fps` into the generated `config.json`, so the downstream prompt, storyboard, and video render stages use the selected resolution and timing rate. Full-Auto accepts FPS values `16`, `24`, and `50`; the default is `24`.

`--video-pipeline` selects the downstream video mode when `--run-video-pipeline` is set and is also written to generated `config.json`. Use `ltx_msr` for MSR/reference-guided rendering or `ltx_i2v` for classic storyboard/start-frame rendering. The CLI default is `ltx_i2v` unless overridden.

When `--run-video-pipeline` is set, `full_auto.py` accepts the same runner override surface as `run_pipeline.py`, including workflow paths, `--render-mode`, single-prompt node settings, LoRA overrides, smoke flags, skip flags, concat flags, and `--rolling-frame-profile`.

Example:

```powershell
uv run python full_auto.py `
  --idea "friendship and joy in a bright city" `
  --style "upbeat contemporary pop, warm, catchy, male vocal" `
  --project-name joy_demo `
  --fps 24 `
  --run-video-pipeline `
  --video-pipeline ltx_i2v `
  --render-mode single_prompt `
  --single-prompt-workflow ./workflows/video_ltxv_i2v_v2.json `
  --storyboard-workflow ./workflows/image/image-model/image_t2i_startframe_v1.json `
  --video-lora-1-strength-model 0.7 `
  --video-lora-1-strength-clip 0.6 `
  --smoke-only `
  --smoke-scene 3 `
  --skip-tests
```

## Runner Parameters

`run_pipeline.py` accepts these parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `project_root` | `./projects/my_first_project` | Positional project folder containing `config.json`. If it points to a file, it is treated as `--project-config`. |
| `--project-config` | derived from `project_root` | Explicit path to project `config.json`. |
| `--app-config` | `./app_config.json` | Global LLM and ComfyUI config. |
| `--concept-batch-size` | `10` | Number of scenes per concept-generation LLM batch. Use `0` to disable batching. |
| `--storyboard-workflow` | `./workflows/image/image-model/image_t2i_startframe_v1.json` | ComfyUI API workflow for Z-Image startframes. |
| `--reference-hero-workflow` | `./workflows/image/image-model/image_t2i_startframe_krea_v1.json` | ComfyUI API workflow for MSR actor/location hero reference images. |
| `--reference-edit-workflow` | `./workflows/image/image-model/image_edit_flux2_klein_1ref_v1.json` | ComfyUI API workflow for MSR actor reference edit views. |
| `--msr-workflow` | `./workflows/video_ltxv_msr_1actor_1background_v4.json` | ComfyUI API workflow used when `--video-pipeline ltx_msr` is selected. |
| `--relay-workflow` | unset | Optional LTX workflow with `#PROMPT_RELAY`; required only for `relay` or `auto`. |
| `--single-prompt-workflow` | `./workflows/video_ltxv_i2v_v2.json` | LTX workflow with single prompt node. |
| `--render-mode` | `single_prompt` | `single_prompt`, `relay`, or `auto`. |
| `--single-prompt-title` | `#PROMPT` | Node title patched for single-prompt LTX. |
| `--single-prompt-input` | `text` | Input field patched on the single-prompt node. |
| `--rolling-frame-profile` | `original` | `original`, `safe`, or `off`. |
| `--storyboard-lora-strength` | unset | Optional storyboard/image `--character-lora-strength`. If unset, the workflow LoRA strength is kept. |
| `--video-character-lora-strength` | unset | Optional LTX `--character-lora-strength`. If unset, the workflow LoRA strength is kept. |
| `--video-lora-1-strength-model` | unset | Optional LTX `--lora-1-strength-model`. If set, it patches the workflow `#LORA_1` model strength even when `lora_1.enabled` is off. |
| `--video-lora-1-strength-clip` | unset | Optional LTX `--lora-1-strength-clip`. If set, it patches the workflow `#LORA_1` CLIP strength even when `lora_1.enabled` is off. |
| `--lora-split-enabled` / `--no-lora-split-enabled` | unset | Optional LTX split override. |
| `--smoke-scene` | `16` | Scene number rendered when `--smoke-only` is set. |
| `--smoke-only` | off | Render only `--smoke-scene` into a smoke output folder. |
| `--no-skip-existing` | off | Re-render existing LTX clips. |
| `--skip-tests` | off | Skip unit tests at the beginning. |
| `--skip-main-pipeline` | off | Skip `main.py`; reuses existing timeline, prompts, and render plan. |
| `--skip-relay-compact` | off | Skip relay prompt compaction. Ignored in `single_prompt` mode. |
| `--skip-anchor-fix` | off | Skip prompt anchor repair. |
| `--skip-storyboard` | off | Skip storyboard generation. |
| `--skip-storyboard-page` | off | Skip static `storyboard/index.html` generation. |
| `--skip-msr-reference-render` | off | For `ltx_msr`, skip ComfyUI reference rendering and reuse existing manifests under `output/references`. The runner still writes the `_refs.json` render plan. |
| `--skip-ltx` | off | Skip LTX rendering. |
| `--skip-final-concat` | off | Skip final FFmpeg concat and mux. |
| `--diagnostic-original-audio-mux` | off | Also create a diagnostic concat using per-scene audio streams. |
| `--no-original-audio-mux` | deprecated | Kept for compatibility; final output always uses original full-song audio. |

Examples:

```powershell
uv run python run_pipeline.py ./projects/my_song --smoke-only --smoke-scene 3
```

```powershell
uv run python run_pipeline.py ./projects/my_song --no-skip-existing --rolling-frame-profile safe
```

```powershell
uv run python run_pipeline.py --project-config ./projects/my_song/config.json --skip-tests
```

If `config.json` has active `loras` entries, the runner passes the project config to `render_ltx.py`; the renderer resolves enabled slots, explicitly configured names, explicitly configured strengths, and split mode from that file. Legacy `lora_1.enabled: true` still works when `loras` is absent.

## LTX MSR Runner

Use `--video-pipeline ltx_msr` to render with Multi Subject Reference instead of storyboard startframes. In this mode the runner automatically:

1. skips storyboard frame rendering and the storyboard HTML page,
2. renders actor and location references into `output/references`,
3. writes `output/render/plans/references.json`,
4. renders MSR clips into `output/render/ltx_msr`,
5. concatenates the trimmed clips and muxes the original full song audio.

MSR frame count and preroll interact. The Licon MSR node consumes a fixed internal reference frame count such as 17 or 41 frames. FeverSlop's default rolling window adds 50 preroll frames before most scene clips. As a rule of thumb, keep the preroll segment longer than the MSR reference frame count. With the default 50-frame preroll, an MSR frame count of 17 is usually safe, 41 can be borderline for short or abrupt scenes, and values around 71 can exceed the preroll and make reference-sheet content visibly leak into the clip. If the workflow MSR frame count is changed, adjust preroll accordingly or inspect the first rendered frames carefully.

Actor MSR references should be generated on a neutral solid background, preferably white or black. This helps the model separate the character from the reference background. Location references are the exception: they should remain real environment/background images because they feed the MSR background input.

Copy the project folder to the render machine with at least `config.json` and `input/<song>.mp3`, make sure `app_config.json`, ComfyUI, the required models, and the workflow JSON files exist there, then run:

```powershell
uv run python run_pipeline.py .\projects\dwarfventure-msr `
  --video-pipeline ltx_msr `
  --skip-tests
```

For a one-scene smoke render:

```powershell
uv run python run_pipeline.py .\projects\dwarfventure-msr `
  --video-pipeline ltx_msr `
  --smoke-only `
  --smoke-scene 1 `
  --skip-tests
```

To reuse already rendered references after changing only the MSR video workflow:

```powershell
uv run python run_pipeline.py .\projects\dwarfventure-msr `
  --video-pipeline ltx_msr `
  --skip-msr-reference-render `
  --no-skip-existing `
  --skip-tests
```

To re-render one or more scenes as new visual variants, keep the existing references and pass `--randomize-seed`. Existing clips are skipped unless `--no-skip-existing` is also set:

```powershell
uv run python run_pipeline.py .\projects\dwarfventure-msr `
  --video-pipeline ltx_msr `
  --skip-msr-reference-render `
  --randomize-seed `
  --no-skip-existing `
  --smoke-only `
  --smoke-scene 4 `
  --skip-tests
```

Override the workflow paths when the render machine uses different workflow filenames:

```powershell
uv run python run_pipeline.py .\projects\dwarfventure-msr `
  --video-pipeline ltx_msr `
  --reference-hero-workflow .\workflows\image\image-model\image_t2i_startframe_krea_v1.json `
  --reference-edit-workflow .\workflows\image\image-model\image_edit_flux2_klein_1ref_v1.json `
  --msr-workflow .\workflows\video_ltxv_msr_1actor_1background_v4.json `
  --skip-tests
```

For a recurring project choice, store the paths in the project config instead
of repeating flags:

```json
{
  "video_pipeline": "minimax-h3-r2v",
  "workflows": {
    "video": "workflows/video/minimax_h3/r2v_eb57_8s_v1.json",
    "reference_hero": "workflows/image/image-model/image_t2i_startframe_krea_v1.json",
    "reference_edit": "workflows/image/image-model/image_edit_flux2_klein_1ref_v1.json"
  }
}
```

The safe CLI hashes the selected workflow files, not only their path. Changing
the contents in place is therefore visible on the next dry-run. Precedence is:
explicit CLI override, project config, then the existing CLI default. The
precedence is applied per field, so an explicit video workflow still keeps
configured reference workflows. The workflow flags remain useful for a one-off
compatibility run and have not been renamed. Project-wide resolution or
workflow changes cannot be resumed with a partial `--scenes` selection; omit
the selector so the final assembly cannot mix incompatible clips.

## app_config.json

`app_config.json` configures infrastructure:

```json
{
  "llm": {
    "api_key": "your-local-key",
    "base_url": "http://localhost:8080/v1",
    "model": "default",
    "models": {},
    "temperature": 0.7,
    "max_tokens": 4096,
    "max_concurrent_requests": 1
  },
  "comfyui": {
    "base_url": "http://127.0.0.1:8188",
    "model_overrides": []
  },
  "storyboard_prompt_transforms": []
}
```

| Key | Default | Meaning |
| --- | --- | --- |
| `llm.api_key` | none | Optional local API key; alternatively use `LLM_API_KEY` in the adjacent `.env`. |
| `llm.base_url` | `http://localhost:8080/v1` | OpenAI-compatible API base URL. |
| `llm.model` | `default` | Model name sent to the LLM server. |
| `llm.models` | `{}` | Optional task-profile model overrides. Missing profiles fall back to `llm.model`. |
| `llm.temperature` | `0.7` | Prompt creativity. Lower is more stable. |
| `llm.max_tokens` | `4096` | Max response length. Increase if JSON is truncated. |
| `llm.max_concurrent_requests` | `1` | Process-local ceiling shared by direct OpenAI-compatible calls and DSPy/LiteLLM calls. |
| `comfyui.base_url` | `http://127.0.0.1:8188` | ComfyUI API URL. |
| `comfyui.model_overrides` | `[]` | Optional strict model-reference overrides for workflow portability edge cases. |
| `storyboard_prompt_transforms` | `[]` | Optional workflow-specific LLM prompt transforms before storyboard rendering. |

`LLM_API_KEY` from the process environment overrides `llm.api_key`; the JSON
value overrides `LLM_API_KEY` from the `.env` beside `app_config.json`. Both
local files are ignored by Git and must not be committed with real credentials.

`llm.max_concurrent_requests` coordinates only LLM calls inside the current
FeverSlop Python process. It is not a server-side lock and does not limit other
FeverSlop processes or external clients using the same LLM server.

Ideogram4 storyboard workflows can opt into a raw LLM prompt transform:

```json
{
  "storyboard_prompt_transforms": [
    {
      "workflow": "workflows/image/image-model/image_t2i_startframe_ideogram_v1.json",
      "kind": "template",
      "template": "documentation/ideogram4_prompt_template.md",
      "positive_prompt_input": "text",
      "debug_dir": "ideogram4_prompt_debug"
    }
  ]
}
```

For `kind: "template"`, FeverSlop sends the template's `[SYSTEM]` section as the system prompt, fills the `[USER]` section with the scene `width`, `height`, and original storyboard prompt, then writes the raw LLM response into the workflow prompt input. No JSON validation is performed.

ComfyUI workflow model references are resolved automatically against the server selected by `comfyui.base_url` before each render request is queued. This keeps `workflows/*.json` portable across Windows/Linux ComfyUI servers without rewriting workflow files.

Detailed behavior, matching rules, overrides, and validation command:

```text
documentation/comfyui_model_resolution.md
```

## config.json

Complete project config shape:

```json
{
  "project_name": "forest_song",
  "input_audio": "input/my_song.mp3",
  "lyrics": "",
  "video": {
    "fps": 24,
    "width": 1280,
    "height": 704
  },
  "audio": {
    "demucs_model": "htdemucs_ft",
    "whisper_model": "large",
    "language": "en"
  },
  "scene_generation": {
    "min_duration": 2.0,
    "max_duration": 10.0,
    "bias": 0.7,
    "duration_preset": "impact_weighted",
    "seed": -1
  },
  "vocal_detection": {
    "merge_gap": 0.5,
    "min_vocal_duration": 0.4,
    "min_silence_duration": 0.8,
    "rms_low_percentile": 20,
    "rms_high_percentile": 85,
    "rms_ratio": 0.35,
    "smooth_frames": 10
  },
  "story_idea": "",
  "style": "",
  "subject": "",
  "locations": [],
  "steering": {
    "global": "",
    "story_idea": "",
    "style": "",
    "subject": "",
    "locations": "",
    "concepts": "",
    "zimage": "",
    "ltx": "",
    "final_prompts": ""
  },
  "prompt_guidance": {
    "character_visibility": "",
    "shot_types": "",
    "environments": "",
    "lighting": "",
    "camera_motion": "",
    "physical_interaction": "",
    "facial_expression": "",
    "outfit_rules": "",
    "prompt_structure": "",
    "list_handling": "",
    "word_count_min": 40,
    "word_count_max": 50
  },
  "lora_1": {
    "enabled": false,
    "name": "",
    "strength_model": 1.0,
    "strength_clip": 1.0
  },
  "lora_split_enabled": false,
  "loras": []
}
```

### Top-Level Keys

| Key | Default | Meaning |
| --- | --- | --- |
| `project_name` | audio filename stem | Human-readable project name. |
| `input_audio` | required | Audio file path, relative to `config.json` or absolute. |
| `lyrics` | empty | Optional complete reference lyrics for correcting detected vocal text without changing timing. |
| `video_pipeline` | `ltx_i2v` from CLI default, or `ltx_msr` when selected | Video rendering pipeline mode. `ltx_i2v` is classic storyboard/start-frame rendering; `ltx_msr` uses actor/location references and MSR workflows. |
| `story_idea` | empty | Hard override for the generated story concept. |
| `style` | empty | Hard override for visual style. |
| `subject` | empty | Hard override for the main character or object. Use this for identity consistency. |
| `locations` | `[]` | Hard list of allowed locations. When non-empty, every concept and Z-Image prompt must visibly use one allowed location or a direct visual variant. |
| `trigger_word` | empty | Optional extra trigger word passed into scene prompt generation. |
| `lora_split_enabled` | `false` | Enables split LoRA patching for active entries in `loras`. |
| `loras` | `[]` | Index-stable LTX LoRA list. Position 1 maps to `#LORA_1`, position 2 to `#LORA_2`, and so on. |

Hard overrides are stronger than steering notes. Use them when something must remain fixed.

### lyrics

Optional complete reference lyrics for the song. When this field is set, FeverSlop still uses Whisper and vocal-energy detection for timing, but corrects the detected vocal segment text against these reference lyrics before building scene prompts.

Use this when Whisper hears the right timing but gets words wrong. Do not use it to force different timing; segment boundaries are preserved.

### video

| Key | Default | Meaning |
| --- | --- | --- |
| `fps` | `24` | Output frame rate used by the default LTX 2.3 workflows. |
| `width` | `1280` | Frame width. Must match the workflow and model constraints. |
| `height` | `704` | Frame height. Must match the workflow and model constraints. |

Frame counts are snapped from absolute scene start/end times:

```text
frame_count = round(end_seconds * fps) - round(start_seconds * fps)
```

This avoids accumulated audio drift across many clips.

### audio

| Key | Default | Meaning |
| --- | --- | --- |
| `demucs_model` | `htdemucs_ft` | Demucs model for stem separation. |
| `whisper_model` | `large` | Whisper model for lyric transcription from the vocal stem. |
| `language` | `en` | Whisper language code, for example `de` or `en`. |

### scene_generation

| Key | Default | Meaning |
| --- | --- | --- |
| `min_duration` | `2.0` | Minimum legal scene duration in seconds. |
| `max_duration` | `10.0` | Maximum legal scene duration in seconds. |
| `bias` | `0.7` | Beat/impact weighting for cut placement. |
| `duration_preset` | `impact_weighted` | Scene duration strategy. |
| `seed` | `-1` | Scene-boundary seed. `-1` picks a new random seed each run; any other integer is reproducible. |

Lower `max_duration` creates faster cutting and more LTX jobs. Higher `max_duration` gives longer shots but can make LTX consistency harder.

### minimax_h3_audio_refs

Configures which audio stems are passed as reference audio to the MiniMax H3 R2V model.
Requires `"video_pipeline": "minimax-h3-r2v"`.

| Key | Default | Meaning |
| --- | --- | --- |
| `stems` | `["vocals", "full_mix"]` | Ordered list of audio stems. `vocals` and `full_mix` are always prioritized for the first two slots (critical for lip-sync). `full_mix` maps to the song's input audio. Valid values: `vocals`, `drums`, `bass`, `other`, `full_mix`. |

```json
{
  "video_pipeline": "minimax-h3-r2v",
  "minimax_h3_audio_refs": {
    "stems": ["vocals", "full_mix", "drums"]
  }
}
```

The MiniMax H3 R2V node has 3 audio reference slots:
- **Slot 0:** Main audio (trimmed to scene duration) — always occupied
- **Slots 1–2:** Stem audio (trimmed to scene duration) — configurable

When `vocals` is in the stem list, it provides the vocal track for improved
lip-sync alignment. When `full_mix` is included, it provides the full song
for beat and rhythm context. These two are always prioritized regardless of
list order.

### vocal_detection

| Key | Default | Meaning |
| --- | --- | --- |
| `merge_gap` | `0.5` | Merge nearby same-kind vocal/instrumental segments. |
| `min_vocal_duration` | `0.4` | Ignore or absorb very short vocal detections. |
| `min_silence_duration` | `0.8` | Avoid treating tiny gaps as real silence. |
| `rms_low_percentile` | `20` | Low RMS percentile for adaptive thresholding. |
| `rms_high_percentile` | `85` | High RMS percentile for adaptive thresholding. |
| `rms_ratio` | `0.35` | Vocal activity threshold ratio. |
| `smooth_frames` | `10` | Smoothing window for vocal activity. |

If the timeline marks too much as singing, increase `min_vocal_duration`, `min_silence_duration`, or `rms_ratio`. If it misses short sung words, reduce them carefully.

### steering

Steering values are soft instructions injected into LLM stages. Top-level `locations` are stronger: they automatically constrain concept and Z-Image prompt generation to the allowed location list.

| Key | Used For |
| --- | --- |
| `global` | General notes included in several global context steps. |
| `story_idea` | Extra notes for story generation. |
| `style` | Extra notes for style generation. |
| `subject` | Extra notes for subject generation. |
| `locations` | Extra notes for location generation; use top-level `locations` for hard allowed-location constraints. |
| `concepts` | Extra notes for per-scene concept prompts. |
| `zimage` | Extra notes for startframe image prompts. |
| `ltx` | Extra notes for I2V motion prompts. |
| `final_prompts` | Reserved for later final prompt-stage steering. |

Use steering for preference, not strict identity. Example:

```json
"steering": {
  "global": "Keep the video grounded and cinematic, with no comedy or surreal props.",
  "zimage": "Prefer medium close-ups where the subject's face and hands are visible.",
  "ltx": "Preserve the provided start frame and use restrained camera movement."
}
```

### prompt_guidance

`prompt_guidance` gives reusable prompt vocabulary to the LLM.

| Key | Meaning |
| --- | --- |
| `character_visibility` | Visibility rules, such as always visible, centered, medium close-up. |
| `shot_types` | Preferred shot types. |
| `environments` | Allowed or preferred environments. |
| `lighting` | Lighting vocabulary. |
| `camera_motion` | Camera motion vocabulary. |
| `physical_interaction` | Safe physical actions that do not change the story. |
| `facial_expression` | Expression vocabulary. |
| `outfit_rules` | Wardrobe continuity rules. |
| `prompt_structure` | Preferred structure for concept prompts. |
| `list_handling` | Notes on how to use guidance lists. |
| `word_count_min` | Lower target for concept prompt length. |
| `word_count_max` | Upper target for concept prompt length. |

### loras and lora_1

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Enables LoRA patching for LTX renders. |
| `name` | empty | Optional LoRA filename as ComfyUI expects it. If empty, the renderer keeps the workflow's existing `lora_name` and patches only strengths. |
| `strength_model` | `1.0` | Model strength. |
| `strength_clip` | `1.0` | CLIP strength, if the node exposes that input. |

Use the new `loras` array for multiple LTX LoRAs:

```json
"lora_split_enabled": true,
"loras": [
  {
    "enabled": true,
    "name": "characters/main.safetensors",
    "strength_model": 0.8,
    "strength_clip": 0.8
  },
  {
    "enabled": false,
    "name": "characters/unused_second_slot.safetensors",
    "strength_model": 0.6,
    "strength_clip": 0.6
  },
  {
    "enabled": true,
    "name": "styles/third_slot.safetensors",
    "strength_model": 0.5,
    "strength_clip": 0.5
  }
]
```

Array positions are stable workflow indices. The first entry maps to `#LORA_1`, the second to `#LORA_2`, and the third to `#LORA_3`; disabled entries are skipped but keep their index. If `loras` is missing, legacy `lora_1` is still read as LoRA 1. If `loras` exists, it wins over `lora_1`. Only properties present in config or CLI are patched; omitted `name`, `strength_model`, and `strength_clip` values keep the workflow defaults.

Without split, active LoRA `N` patches:

```text
#LORA_N
```

With `lora_split_enabled: true`, active LoRA `N` uses both anchors when both are present:

```text
#LORA_N
#SPLIT_LORA_N
```

For explicitly set strengths, the renderer writes half strength to `#LORA_N` and full strength to `#SPLIT_LORA_N` when the split anchor exists. If `#SPLIT_LORA_N` is missing, `#LORA_N` receives full strength even when split is enabled. If split is disabled and both anchors exist, both receive full strength. If `name` is set, it also writes that LoRA filename to the patched nodes. If `name` is empty or omitted, it leaves each workflow node's existing `lora_name` untouched. The code only patches existing nodes. It does not insert or wire LoRA nodes.

## Manual Command Order

The runner is easiest, but the manual order is useful for debugging.

### 1. Main Pipeline

```powershell
uv run python main.py `
  --project ./projects/my_song/config.json `
  --app-config ./app_config.json `
  --concept-batch-size 10
```

Outputs:

```text
output/stems/
output/timeline/timeline_<song>.json
output/timeline/beat_data_<song>.json
output/timeline/scenes_<song>_raw.srt
output/timeline/scenes_<song>.srt
output/timeline/stage1_segments_<song>.json
output/prompts/ltx_prompt_relay_<song>.json
output/prompts/resolved_context_<song>.json
output/prompts/concept_prompts_<song>.json
output/prompts/scene_details_<song>.json
output/prompts/scene_prompts_<song>.json
output/render/plans/base.json
```

`main.py` options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--project` | required | Path to project `config.json`. |
| `--app-config` | `app_config.json` | Path to app config. |
| `--render-storyboard` | off | Also render storyboard from inside `main.py`. Usually skip this and call `render_storyboard.py` after anchor fixes. |
| `--zimage-workflow` | none | Required only with `--render-storyboard`. |
| `--concept-batch-size` | `0` | LLM batch size for concept prompts. Use `10` for normal runs. |

### 2. Optional Relay Compaction

Only needed for relay mode or if you want compact relay prompts in the render plan:

```powershell
uv run python compact_relay_prompts.py `
  --app-config ./app_config.json `
  --input-render-plan ./projects/my_song/output/render/plans/base.json `
  --output-render-plan ./projects/my_song/output/render/plans/compact.json `
  --max-words 28
```

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--app-config` | `./app_config.json` | LLM config. |
| `--input-render-plan` | required | Source render plan. |
| `--output-render-plan` | required | Compacted render plan. |
| `--max-words` | `28` | Target max words per compacted relay prompt. |

### 3. Prompt Anchor Fix

Recommended before storyboard and LTX:

```powershell
uv run python fix_ltx_prompt_anchors.py `
  --input-render-plan ./projects/my_song/output/render/plans/compact.json `
  --output-render-plan ./projects/my_song/output/render/plans/anchored.json `
  --subject-anchor "the same old druid shaman man, weathered face, long grey beard, rough linen robe"
```

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--input-render-plan` | required | Source render plan. |
| `--output-render-plan` | required | Anchor-fixed output plan. |
| `--subject-anchor` | required | Stable subject phrase prepended to prompts. |
| `--max-base-prompt-chars` | `1200` | Max base prompt length after anchoring. |
| `--max-relay-chars` | `260` | Max relay prompt length after anchoring. |

### 4. Storyboard Startframes

```powershell
uv run python render_storyboard.py `
  --app-config ./app_config.json `
  --render-plan ./projects/my_song/output/render/plans/anchored.json `
  --workflow ./workflows/image/image-model/image_t2i_startframe_v1.json `
  --output-dir ./projects/my_song/output/render/storyboard
```

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--app-config` | `./app_config.json` | ComfyUI config. |
| `--render-plan` | required | Render plan to use. Must match the later LTX plan. |
| `--workflow` | required | Z-Image API workflow JSON. |
| `--output-dir` | required | Output folder for `scene_XXXX.png`. |
| `--limit` | none | Render only the first N scenes. |
| `--scenes` | none | Scene selector, for example `1,2,5-8`. |
| `--no-skip-existing` | off | Re-render existing frames. |
| `--character-lora-strength` | workflow value | Optional strength override for storyboard `#LORA_1`. If omitted, the workflow value is kept. |
| `--negative-prompt` | empty | Extra negative prompt text. |
| `--positive-title` | `#PROMPT_POSITIVE` | Positive prompt node title. |
| `--negative-title` | `#PROMPT_NEGATIVE` | Negative prompt node title. |
| `--save-title` | `#SAVE_IMAGE` | Save-image node title. |
| `--character-lora-title` | `#LORA_1` | Storyboard LoRA node title. |

### 5. Static Storyboard Review Page

Generate an offline HTML page after the storyboard frames exist:

```powershell
uv run python storyboard_page.py `
  --render-plan ./projects/my_song/output/render/plans/anchored.json `
  --storyboard-dir ./projects/my_song/output/render/storyboard `
  --output-html ./projects/my_song/output/render/storyboard/index.html `
  --title "Storyboard Review"
```

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--render-plan` | required | Final render plan used for storyboard and LTX. |
| `--storyboard-dir` | required | Folder containing `scene_XXXX.png`. |
| `--output-html` | `storyboard/index.html` | Static HTML output path. |
| `--title` | `Storyboard Review` | Browser and page title. |
| `--limit` | none | Show only the first N scenes. |
| `--scenes` | none | Scene selector, for example `1,2,5-8`. |
| `--allow-missing-images` | off | Render placeholders instead of failing when images are missing. |

The page uses `genesis-DESIGN.md` styling and writes relative image links, so the `storyboard/` folder can be moved as a self-contained review artifact. Each scene image opens in a new tab when clicked.

View modes:

- `Full`: scene heading, image, story caption, lyrics, motion notes, and collapsible prompt details.
- `Compact`: image-only grid for visual review; the Z-Image prompt is available as the native browser tooltip on the image.

The layout uses the full page width and limits the grid to at most five cards per row so the images do not become too small.

The Python runner generates this page automatically at:

```text
output/render/storyboard/index.html
```

Use `--skip-storyboard-page` to skip only this HTML export.

### 6. LTX Smoke Scene

Render one scene before a full run:

`render_ltx.py` can read project defaults directly when `--project-config` is set. If you omit it, the script also tries to discover `config.json` by walking upward from `--render-plan`, which covers the normal `project/output/render/...` layout. That keeps scene-duration and LoRA values in `config.json` while still allowing explicit CLI overrides.

```powershell
uv run python render_ltx.py `
  --project-config ./projects/my_song/config.json `
  --app-config ./app_config.json `
  --render-plan ./projects/my_song/output/render/plans/anchored.json `
  --workflow ./workflows/video_ltxv_i2v_v2.json `
  --render-mode single_prompt `
  --audio ./projects/my_song/input/my_song.mp3 `
  --storyboard-dir ./projects/my_song/output/render/storyboard `
  --output-dir ./projects/my_song/output/render/ltx_single_prompt_smoke `
  --scenes 1 `
  --no-skip-existing `
  --debug-workflows-dir ./projects/my_song/output/render/ltx_single_prompt_debug
```

Inspect:

```text
output/render/ltx_single_prompt_smoke/final/scene_0001.mp4
output/render/ltx_single_prompt_debug/scene_0001_workflow.json
```

### 6. Full LTX Render

```powershell
uv run python render_ltx.py `
  --project-config ./projects/my_song/config.json `
  --app-config ./app_config.json `
  --render-plan ./projects/my_song/output/render/plans/anchored.json `
  --workflow ./workflows/video_ltxv_i2v_v2.json `
  --render-mode single_prompt `
  --audio ./projects/my_song/input/my_song.mp3 `
  --storyboard-dir ./projects/my_song/output/render/storyboard `
  --output-dir ./projects/my_song/output/render/ltx_single_prompt `
  --debug-workflows-dir ./projects/my_song/output/render/ltx_single_prompt_debug
```

`render_ltx.py` options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--app-config` | `./app_config.json` | ComfyUI config. |
| `--project-config` | none | Optional project config source for scene-duration and LoRA defaults. |
| `--render-plan` | required | Render plan JSON. |
| `--workflow` | required | Main LTX workflow. Relay mode uses this as relay workflow. Single-prompt mode uses this unless `--single-prompt-workflow` is set. |
| `--render-mode` | `single_prompt` | `single_prompt`, `relay`, or `auto`. |
| `--single-prompt-workflow` | none | Required for `auto`; optional override for single-prompt scenes. |
| `--single-prompt-title` | `#PROMPT` | Single prompt node title. |
| `--single-prompt-input` | `text` | Single prompt node input field. |
| `--audio` | required | Original song audio. |
| `--storyboard-dir` | required | Folder containing `scene_XXXX.png` startframes. |
| `--output-dir` | required | LTX output directory. |
| `--limit` | none | Render first N scenes. |
| `--scenes` | none | Scene selector such as `1,2,5-8`. |
| `--no-skip-existing` | off | Re-render existing final clips. |
| `--character-lora-strength` | workflow value | Optional legacy CLI name for overriding video `#LORA_1` strength. If omitted, the workflow value is kept. |
| `--lora-1-enabled` | off | Enable LoRA slot 1. Only explicitly supplied name/strength flags are patched. |
| `--lora-1-name` | empty | Optional LoRA filename override for `#LORA_1`. If omitted, the workflow's existing `lora_name` is kept. |
| `--lora-1-strength-model` | `1.0` | Model strength for `#LORA_1`. |
| `--lora-1-strength-clip` | `1.0` | CLIP strength for `#LORA_1`. |
| `--lora-split-enabled` | project config or off | Patch `#LORA_N` with half strength and `#SPLIT_LORA_N` with full strength for active LoRAs. |
| `--randomize-seed` | off | Randomize seed per render. |
| `--seed-offset` | `100000` | Deterministic seed offset. |
| `--no-upload-audio` | off | Reuse an already uploaded ComfyUI audio file. |
| `--uploaded-audio-name` | none | Name of the uploaded audio file when upload is disabled. |
| `--no-upload-startframes` | off | Reuse startframes already available to ComfyUI. |
| `--segment-length-mode` | `frames_minus_one` | Relay segment sum mode: `frames_minus_one` or `frames`. |
| `--min-duration` | `2.0` | Validation minimum scene duration. |
| `--max-duration` | `10.0` | Validation maximum scene duration. |
| `--allow-out-of-range-clips` | off | Do not fail on scene duration validation. |
| `--debug-workflows-dir` | none | Save patched workflow JSONs for inspection. |
| `--rolling-frame-profile` | `original` | `original`, `safe`, or `off`. |
| `--preroll-frames` | profile value | Override profile preroll frames. |
| `--tail-loss-frames` | profile value | Override profile tail frames. |
| `--no-postprocess` | off | Skip clip trimming/muxing after ComfyUI render. |
| `--ffmpeg` | `ffmpeg` | FFmpeg executable path. |
| `--postprocess-streamcopy` | off | Use streamcopy instead of re-encoding during postprocess. |

If `--project-config` is set, `render_ltx.py` uses `scene_generation.min_duration`, `scene_generation.max_duration`, `loras`, `lora_split_enabled`, and legacy `lora_1.*` from that file unless you override them on the command line.

### 7. Final Concat and Audio Mux

The runner does this automatically. Manual equivalent:

```powershell
ffmpeg -y `
  -f concat -safe 0 `
  -i ./projects/my_song/output/render/ltx_single_prompt/concat_list.txt `
  -an -c:v copy `
  ./projects/my_song/output/render/ltx_single_prompt/my_song_video_only.mp4

ffmpeg -y `
  -i ./projects/my_song/output/render/ltx_single_prompt/my_song_video_only.mp4 `
  -i ./projects/my_song/input/my_song.mp3 `
  -map 0:v:0 -map 1:a:0 `
  -c:v copy -c:a aac -b:a 320k -shortest `
  ./projects/my_song/output/render/ltx_single_prompt/my_song.mp4
```

This avoids per-scene audio splice hiccups. The video clips are concatenated without audio, then the original full song is muxed once.

## Optional: Project Archive

Use this when a project has produced enough intermediate files that you want one portable artifact before manual cleanup:

```powershell
uv run python -m tools.project_asset_archive --project ./projects/my_song/config.json
```

The archive includes project-local working files and writes `archive_manifest.json` inside the ZIP. It excludes the selected project config, `output/render/storyboard/**`, final muxed videos such as `final_concat.mp4` or `<project_name>.mp4`, and `archives/` itself. The command is non-destructive; it does not delete or move source files.

## Recommended Workflow Choices

Use `single_prompt` unless you specifically need frame-level prompt changes inside a scene. The single-prompt path follows the default FeverSlop quality pattern:

```text
concept prompt -> Z-Image startframe prompt -> startframe -> I2V prompt based on that startframe prompt
```

Use `relay` only when:

- the workflow contains a correctly wired `#PROMPT_RELAY` node
- a scene needs internal singing/silent prompt changes
- you have already validated the relay debug workflow JSON

Use `auto` only when both workflows are working:

```powershell
--workflow ./workflows/your_prompt_relay_workflow.json `
--single-prompt-workflow ./workflows/video_ltxv_i2v_v2.json `
--render-mode auto
```

## Rolling Frame Profiles

| Profile | Preroll | Tail | 8N+1 rounding | Use Case |
| --- | ---: | ---: | --- | --- |
| `original` | `50` | `25` | yes | Default high-quality rolling-frame behavior. |
| `safe` | `6` | `0` | no | Lower VRAM, faster debugging. |
| `off` | `0` | `0` | no | Maximum simplicity, less transition help. |

The final clips are trimmed back to the exact scene frame count. Audio alignment comes from absolute frame-snapped scene boundaries and final original-song muxing.

## Steering Tutorial

### Global Fields vs Steering vs Prompt Guidance

There are three levels of control. Use the strongest level only where you need it.

| Layer | Config Keys | Strength | Best For |
| --- | --- | --- | --- |
| Global hard fields | `story_idea`, `style`, `subject`, `locations` | Strongest | Fixed identity, fixed world, fixed visual premise. |
| Steering | `steering.global`, `steering.concepts`, `steering.zimage`, `steering.ltx` | Medium | Soft instructions, emphasis, avoidance rules, stage-specific nudges. |
| Prompt guidance | `prompt_guidance.*` | Vocabulary rails | Lists of shot types, lighting, expressions, environment motifs, outfit rules. |

Use global hard fields for what the video **is**:

```json
{
  "story_idea": "A lonely ritual performance where an old shaman sings to awaken a forgotten forest shrine.",
  "style": "dark cinematic fantasy, natural textures, realistic faces, restrained magical atmosphere",
  "subject": "the same old druid shaman man, weathered scarred face, long grey beard, rough linen robe, carved wooden staff",
  "locations": [
    "ancient mossy forest shrine",
    "foggy stone circle beneath twisted trees"
  ]
}
```

When `locations` is non-empty, the pipeline injects an allowed-location constraint into concept generation and Z-Image prompt generation. Each scene concept and startframe prompt must visibly take place in one listed location or a direct visual variant, and the LLM is instructed not to invent other locations.

Use steering for what the generator should **emphasize or avoid**:

```json
{
  "steering": {
    "global": "Keep the whole video serious, grounded, and emotionally intimate.",
    "concepts": "Do not introduce new main characters, modern props, logos, vehicles, or unrelated story events.",
    "zimage": "Keep compositions simple: one readable subject, clear face, readable hands, and no clutter.",
    "ltx": "Preserve the provided start frame. Use slow cinematic motion and avoid sudden pose, outfit, or location changes."
  }
}
```

Use prompt guidance for the ingredients the LLM may **choose from**:

```json
{
  "prompt_guidance": {
    "character_visibility": "The subject must remain visible and clearly framed; face and hands should usually be readable.",
    "shot_types": "medium close-up, low-angle close-up, shoulder-level tracking shot, slow push-in",
    "lighting": "moonlight, candlelight, soft green forest rim light, faint mist diffusion",
    "camera_motion": "slow forward dolly, subtle handheld drift, restrained orbit",
    "facial_expression": "grief, trance, quiet resolve, haunted concentration",
    "outfit_rules": "Keep the same robe, staff, beard, facial features, and rough natural materials in every scene."
  }
}
```

Practical rule:

- Put fixed identity and fixed world facts in top-level fields.
- Put "do more of this" and "avoid that" in `steering`.
- Put reusable lists and visual vocabulary in `prompt_guidance`.

If you omit `prompt_guidance`, the pipeline still works. The LLM receives fewer custom vocabulary rails, and the built-in LTX detail lists still add deterministic motion, lighting, mood, atmosphere, and expression variation later.

### Strong Character Consistency

Use hard config fields:

```json
{
  "subject": "the same old druid shaman man, weathered scarred face, long grey beard, rough linen robe, carved wooden staff",
  "locations": [
    "ancient mossy forest shrine",
    "foggy stone circle beneath twisted trees"
  ],
  "prompt_guidance": {
    "character_visibility": "The subject must remain visible and clearly framed in every scene.",
    "outfit_rules": "Keep the same robe, staff, beard, and facial features throughout."
  }
}
```

### More Cinematic Motion

Use soft steering:

```json
"steering": {
  "ltx": "Use slow cinematic motion, restrained camera drift, natural breathing, and no sudden location changes."
}
```

### Fewer Random Objects

Constrain both concepts and image prompts:

```json
"steering": {
  "concepts": "Do not introduce new characters, animals, vehicles, weapons, logos, or modern objects.",
  "zimage": "Keep the frame simple: one subject, one readable environment, no extra props unless already specified."
}
```

### More Close-Ups

```json
"prompt_guidance": {
  "shot_types": "medium close-up, close-up, shoulder-level tracking shot",
  "character_visibility": "Face and hands should usually be visible; avoid tiny distant subjects."
}
```

### Better Lip-Sync Scenes

```json
"steering": {
  "ltx": "For vocal scenes, keep the face unobstructed and the mouth visible. For instrumental scenes, keep the mouth relaxed and still."
}
```

The code already avoids adding singing/lip-sync language to instrumental-only scenes.

### LoRA Slots

Add `#LORA_N` nodes to the LTX workflow, wire them into the model/clip path, then set:

```json
"lora_split_enabled": true,
"loras": [
  {
    "enabled": true,
    "name": "characters/my_character.safetensors",
    "strength_model": 0.85,
    "strength_clip": 0.65
  }
]
```

When split is enabled, also add the matching `#SPLIT_LORA_N` node. Run one smoke scene with `--debug-workflows-dir` and confirm the debug JSON contains the expected patched `#LORA_N` and `#SPLIT_LORA_N` values.

## Workflow Anchor Requirements

The code patches ComfyUI API workflows by `_meta.title`.

Common anchors:

| Anchor             | Purpose |
|--------------------| --- |
| `#PROMPT`          | Single-prompt LTX positive prompt. |
| `#PROMPT_RELAY`    | Relay prompt node. |
| `#LORA_N`          | LTX LoRA slot `N`. |
| `#SPLIT_LORA_N`    | Optional split-model LTX LoRA slot `N`; when present it receives full explicit strength. |
| `#PROMPT_POSITIVE` | Storyboard positive prompt default. |
| `#PROMPT_NEGATIVE` | Storyboard negative prompt default. |
| `#SAVE_IMAGE`      | Storyboard save node default. |
| `#LORA_1`          | First LoRA slot; storyboard and video workflows use this as the default LoRA anchor. |

When exporting a new workflow:

1. Export the ComfyUI API workflow JSON.
2. Add stable `_meta.title` values for dynamic nodes.
3. Wire LoRA nodes yourself; the code does not insert nodes.
4. Run a smoke scene with `--debug-workflows-dir`.
5. Inspect the patched debug workflow JSON.

## Debugging

If audio drifts or has clip-boundary hiccups:

1. Confirm `<project_name>.mp4` was made by video-only concat plus original audio mux.
2. Confirm the render plan was generated after the frame-lock fix.
3. Check that sum of `frame_count` matches the expected full-song frame timeline.

If LTX ignores the startframe:

1. Make sure storyboard and LTX use the same final render plan.
2. Run `fix_ltx_prompt_anchors.py`.
3. Inspect `ltx.original_style_i2v_prompt` and confirm it starts from the `z_image.prompt`.
4. Inspect the saved debug workflow.

If ComfyUI validation fails:

1. Check the exact missing anchor in the error.
2. Open the workflow JSON and verify `_meta.title`.
3. Do not rely on node labels visible in the UI unless they are exported into `_meta.title`.

If LLM JSON is truncated:

1. Increase `llm.max_tokens`.
2. Use `--concept-batch-size 5` or `10`.
3. Lower prompt verbosity in `prompt_guidance`.

If output is too expensive or slow:

1. Use `--smoke-only`.
2. Use `--rolling-frame-profile safe`.
3. Use `render_ltx.py --limit 3`.
4. Keep storyboard frames and re-render only selected LTX scenes with `--scenes`.

## Editable Timeline Exports

After rendered scene clips and the render plan already exist, export an
editable timeline without running the upstream pipeline again:

```powershell
uv run python run_pipeline.py projects/my_project --stage export_timeline --format mlt
uv run python run_pipeline.py projects/my_project --stage export_timeline --format openshot
```

Without `--format`, both projects are generated.

The MLT project is written below `output/timeline/` and can be opened by
Shotcut or Kdenlive. It contains the rendered clips in render-plan order and
the original input audio on a separate track. The OpenShot `.osp` export is
kept for compatibility and experimentation, but is considered experimental
and is not actively maintained. MLT preserves gaps in the render plan; true
overlapping clips and editor-specific logo/title objects are not yet exported.

## Verification Commands

Run tests:

```powershell
uv run python -m unittest discover -s tests
```

Check tracked files:

```powershell
git status --short
```
# Current video workflow default

The maintained default is LTX 2.5 Draft with two-pass rendering. Canonical
workflow paths are resolved from `render_profile` (for example,
`ltx25-i2v-draft`) and live below `workflows/video/ltx_25/`. Legacy LTX paths
in older examples must be treated as migration inputs, not as fallback paths.
