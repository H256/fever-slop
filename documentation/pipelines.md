# Pipelines

FeverSlop has two main ways to create a music-video project:

- **Standard music video pipeline**: starts from an existing project folder, `config.json`, and input audio.
- **Full-Auto pipeline**: starts from a project idea and song style, generates ACE-Step audio, scaffolds a project, then optionally runs the video pipeline.

Studio also supports a **Movie project pipeline**. Movie projects start from
prose or screenplay text, create a story arch and cinematic render plan, then
can run a movie full-auto job.

Video rendering can use one of three pipelines:

- **Classic mode**: `ltx_i2v`, storyboard/start-frame driven.
- **MSR mode**: `ltx_msr`, actor/location reference-sheet driven.
- **Ingredients mode**: `ltx_ingredients`, composes per-scene ingredients sheets from actor/location references, injects audio latent from the song, and renders via an ingredients workflow.

Rendering within those pipelines can use one of three render modes:

- **`single_prompt`**: patches `#PROMPT` with the full prompt text. Simplest and
  fastest; default for most music-video scenes.
- **`relay`**: patches `#PROMPT_RELAY` with a structured prompt-relay payload that
  carries multi-scene continuity. Required when the render plan contains a
  multi-state relay (multi-verse, multi-chorus, etc.).
- **`auto`**: selects per-scene from the `render_mode_hint` computed by the render
  plan builder. Scenes with a single relay state or no relay get `single_prompt`;
  scenes with multiple relay states or a `mixed` type get `relay`. Requires both
  `--single-prompt-workflow` and `--relay-workflow` to be configured.

### One-machine LLM/ComfyUI workflow

For `minimax-h3-r2v`, prompt generation is intentionally split around reference
generation. The first prompt stage creates the timeline, scene prompts, and
backend-neutral `subject_directives`. The H3-specific prompt stage runs only
after reference sheets exist, because it must know the actual reference paths
and `<Picture N>` ordering.

This makes the following workflow suitable for a machine on which the LLM and
ComfyUI models cannot remain loaded at the same time:

```text
LLM:  main_pipeline
ComfyUI: msr_references + msr_reference_sheets
LLM:  h3_prompts
ComfyUI: render_plan + ltx_render_scenes
```

The recommended safe CLI can enforce those process boundaries. Set
`execution.vram_handoff` to `manual` in the machine's `app_config.json`, inspect
the full plan, and repeat the same resume command after every printed unload/
load instruction:

```powershell
uv run python main.py run PROJECT --dry-run
uv run python main.py run PROJECT --resume
# unload/load exactly as printed
uv run python main.py run PROJECT --resume
```

Canonical artifacts and scene checkpoints are the handoff state, so no extra
cursor is written. The default `continuous` mode runs without these stops.

The explicit commands below remain an advanced compatibility workflow. They
must be run as separate processes, and their model lifecycle is entirely the
operator's responsibility:

```powershell
# 1. LLM: timeline, scene prompts, and subject/action directives
uv run python run_pipeline.py .\projects\my-song `
  --app-config .\app_config.json `
  --video-pipeline minimax-h3-r2v `
  --stage main_pipeline `
  --skip-tests

# Unload the LLM model and load the ComfyUI reference workflows.

# 2. ComfyUI: actor/location reference generation and reference sheets
uv run python run_pipeline.py .\projects\my-song `
  --app-config .\app_config.json `
  --video-pipeline minimax-h3-r2v `
  --stage msr_references `
  --stage msr_reference_sheets `
  --skip-tests

# Unload ComfyUI models and load the LLM again.

# 3. LLM: compose the reference-aware MiniMax H3 prompts
uv run python run_pipeline.py .\projects\my-song `
  --app-config .\app_config.json `
  --video-pipeline minimax-h3-r2v `
  --stage h3_prompts `
  --skip-tests

# Unload the LLM and load the ComfyUI video workflow again.

# 4. Build the final render plan, then render the scenes
uv run python run_pipeline.py .\projects\my-song `
  --app-config .\app_config.json `
  --video-pipeline minimax-h3-r2v `
  --stage render_plan `
  --skip-tests

uv run python run_pipeline.py .\projects\my-song `
  --app-config .\app_config.json `
  --video-pipeline minimax-h3-r2v `
  --stage ltx_render_scenes `
  --skip-tests
```

The `h3_prompts` stage cannot be moved before reference generation in R2V
mode: its output contains reference labels that depend on the generated
reference artifacts. The backend-neutral subject/action planning itself does
run in `main_pipeline`, so the expensive general prompting work is completed
before the first ComfyUI phase.

## Available Pipeline Entry Points

| Entry point | Use |
| --- | --- |
| `uv run python run_pipeline.py <project>` | Standard end-to-end music-video runner for an existing project. |
| `uv run python full_auto.py ...` | Creates a project from idea/style, renders ACE-Step audio, and can run the video pipeline. |
| `uv run python main.py --project <config.json>` | Lower-level audio/timeline/prompt/render-plan stage, useful for debugging. |
| Studio **Pipeline** page | Starts the same pipeline actions as background jobs and streams progress/logs. |
| Studio `/projects/new/movie` page | Creates movie scaffolds and can start the movie full-auto job. |

`run_pipeline.py` also accepts `--stage <stage>` to run one or more atomic stages
without composing skip flags manually. Example:

```bash
uv run python run_pipeline.py ./projects/my-song \
  --stage anchor_fix
```

LLM-backed prompt stages use the shared `llm.max_concurrent_requests` setting
from `app_config.json`. The default `1` applies across direct
OpenAI-compatible requests and DSPy/LiteLLM requests in the same FeverSlop
process. It is not a server-wide lock; separate FeverSlop processes and other
clients still need server-side/deployment-level concurrency controls.

Ingredients mode example:

```bash
uv run python run_pipeline.py ./projects/my-song \
  --video-pipeline ltx_ingredients \
  --skip-tests
```

### MiniMax H3 Stem Audio References

When using the MiniMax H3 R2V pipeline, you can configure which audio stems
are passed as reference audio clips. The `full_mix` stem always maps to the
song's input audio. The `vocals` stem provides the vocal track split by
Demucs for improved lip-sync quality.

The first two stems are always prioritized (`vocals` then `full_mix`) for
optimal lip-sync. Additional stems fill remaining slots.

```bash
uv run python run_pipeline.py ./projects/my_song \
  --video-pipeline minimax-h3-r2v \
  --minimax-audio-ref-stems vocals,drums,bass
```

Or configure in `config.json`:

```json
{
  "video_pipeline": "minimax-h3-r2v",
  "minimax_h3_audio_refs": {
    "stems": ["vocals", "full_mix"]
  }
}
```

**Valid stem values:** `vocals`, `drums`, `bass`, `other`, `full_mix`.

**Default:** `["vocals", "full_mix"]` — vocals for lip-sync, full mix for beat and rhythm context.

**Note:** `vocals` and `full_mix` are always prioritized for the first two
audio reference slots, regardless of list order. Up to 3 audio reference
slots exist on the MiniMax H3 R2V node; the first is reserved for the main
(comfy) audio, leaving slots 1–2 for stem audio.

For scenes with multiple visible performers, an explicit optional binding can
be stored in the scene references. It is validated before prompt generation
and is serialized into the audio definition without changing slot numbering:

```json
{
  "actor_ids": ["singer", "drummer"],
  "audio_subject_bindings": {
    "vocals": {"subject_id": "singer", "speaker_id": "S1"},
    "drums": {"subject_id": "drummer"}
  }
}
```

The vocal binding requires a speaker ID; `full_mix` deliberately cannot be
bound to a subject because it remains a global beat/rhythm reference. Missing
bindings remain explicitly unbound rather than being inferred.

## Standard Pipeline

CLI:

```bash
uv run python run_pipeline.py ./projects/my-song --skip-tests
```

Studio action:

```text
Project -> Pipeline -> Full pipeline
```

Normal full pipeline stages:

1. Optional unit tests.
2. Main audio/timeline/prompt/render-plan generation.
3. Optional relay prompt compaction.
4. Prompt anchor fixing.
5. Storyboard rendering, classic mode only.
6. Storyboard HTML generation, classic mode only.
7. MSR reference rendering and enrichment, MSR mode only.
8. Ingredients scene sheet composition, Ingredients mode only.
9. LTX video rendering.
10. Final video-only concat.
11. Original full-song audio mux.

### SeedVR2 video upscale

SeedVR2 is an optional post-render stage. Enable it per project in
`config.json` or for a run with `--upscale`:

```json
{
  "upscale": {
    "enabled": true,
    "target_width": 3840,
    "model": "seedvr2_3b_int8_convrot.safetensors",
    "denoise": 0.35,
    "temporal_overlap": 4,
    "color_correction": "lab"
  }
}
```

Set `target_width`, `target_height`, or both. If only one is set, the missing
dimension is derived from the source aspect ratio; if both are set, their
aspect ratio must remain within 2% of the source. If neither is set, the
default is a 2x target. `strategy: "auto"` splits large jumps into bounded passes (maximum
2x per pass and three AI passes by default), which limits VRAM use and reduces
identity drift compared with one extreme pass. The planner rejects downscales
and targets that exceed the configured pass budget.

The default workflow uses the native SeedVR2 3B INT8 ConvRot model. The 7B
variant is intentionally not the default because it can exceed 32 GB VRAM on
the Radeon AI PRO R9700. SeedVR2's current ComfyUI workflow uses fixed model
conditioning; render-plan prompts are therefore recorded as metadata only and
are not sent as text guidance. The API workflow follows ComfyUI's memory-safe
video template: it uses `VAEEncodeTiled` and `VAEDecodeTiled` with 512px tiles,
128px overlap, temporal size 64, and temporal overlap 8 by default. The
temporal tile can be overridden with `vae_temporal_size`. It also preserves the
source video's audio, FPS, and bit depth through `GetVideoComponents` and
`CreateVideo`. Temporal latent split/merge is enabled by default for production
runs so the sampler does not receive the complete clip latent at once. It can
be disabled with `upscale.split_latent: false`, but that increases OOM risk at
large resolutions. The SeedVR2 temporal overlap remains configurable through
the project setting. The effective value is printed per pass and stored in the
scene manifest.

Long clips are additionally processed in short nested segments when a single
resolution pass would exceed the configured memory budget. Set
`upscale.segment_duration_seconds` to the reference duration for the final
resolution (default: 4 seconds). Earlier, lower-resolution passes receive a
proportionally longer cap; this avoids unnecessary concat work while keeping
the final pass conservative. Each segment uses the workflow's native `Video
Slice` path, writes a resumable `upscale_pass_XX_segment_YYYY.mp4` artifact,
and is concatenated before the next resolution pass starts. The start time,
duration, and completion of every segment are logged, and the concat list is
kept beside the pass artifacts for diagnosis.

The stage chooses `final_facefix.mp4` when available, otherwise `final.mp4`,
and writes one resumable artifact per scene:

```text
output/render/scenes/scene_0001/upscale_pass_01.mp4
output/render/scenes/scene_0001/upscale_final.mp4
output/render/scenes/scene_0001/upscale_manifest.json
```

Existing `upscale_final.mp4` files are reused unless `--no-skip-existing` is
passed. Concat prefers `upscale_final.mp4` when present and writes
`movie_upscaled.mp4`; original and FaceFix outputs remain intact.

Timeline export writes independent projects when the corresponding artifacts
exist: `<project>.osp`/`.mlt`, `<project>_facefix.osp`/`.mlt`, and
`<project>_upscaled.osp`/`.mlt`. By default, the export writes both the
Shotcut-compatible MLT project and the OpenShot project. Use
`--stage export_timeline --format mlt` or `--format openshot` to write only one
format.

Important CLI options:

| Option | Meaning |
| --- | --- |
| `--video-pipeline ltx_i2v` | Classic mode. This is the CLI runner default. |
| `--video-pipeline ltx_msr` | MSR mode. Uses reference manifests/sheets and MSR workflow. |
| `--video-pipeline ltx_ingredients` | Ingredients mode. Composes per-scene ingredients reference sheets and renders with an ingredients workflow. |
| `--skip-tests` | Skip the initial unit-test run. Studio pipeline jobs set this internally. |
| `--skip-main-pipeline` | Reuse existing timeline/prompts/render plan. |
| `--scenes 3,5-8` | Render selected scenes. |
| `--smoke-only --smoke-scene 3` | Render one smoke scene into smoke output folders. |
| `--no-skip-existing` | Re-render clips even if outputs already exist. |
| `--stage upscale` | Run only the SeedVR2 upscale stage; respects `upscale.enabled` in the project config. |
| `--upscale` | Temporarily enable the configured SeedVR2 stage for this run. Redundant when the project already has `upscale.enabled: true`. |
| `--upscale-resolution WIDTHxHEIGHT` | Override the project target resolution for this run. |
| `--skip-msr-reference-render` | In MSR mode, reuse existing reference manifests/images. |
| `--skip-msr-prompt-enrichment` | In MSR mode, reuse existing MSR prompt fields. |
| `--skip-ingredients-sheets` | In Ingredients mode, reuse existing ingredients sheets. |
| `--ingredients-workflow PATH` | In Ingredients mode, path to the ingredients ComfyUI workflow JSON (default: `workflows/video_ltxv_ingredients_audio_2stage_v6.json`). |
| `--render-mode single_prompt` | Use `#PROMPT` anchor for every scene. Default. |
| `--render-mode relay` | Use `#PROMPT_RELAY` anchor for every scene. Requires `--relay-workflow`. |
| `--render-mode auto` | Pick per-scene from `render_mode_hint`. Requires `--single-prompt-workflow` and `--relay-workflow`. |
| `--rolling-frame-profile original` | Preroll 50 frames, tail 25 frames, round to 8x. Default. |
| `--rolling-frame-profile safe` | Preroll 6 frames, tail 0 frames, no rounding. Lower VRAM. |
| `--rolling-frame-profile off` | No preroll, no tail, no rounding. Minimal overhead. |
| `--lora-split-enabled` | Split LoRA strength: halve the base LoRA weight and add a second anchor at full strength when the workflow provides a split node. |
| `--no-lora-split-enabled` | Explicitly disable LoRA split. |
| `--randomize-seed` | Use a random seed for each scene instead of scene-number-based seeds. |

## Full-Auto Pipeline

Full-Auto creates a project from:

- `idea`
- `song_style`
- desired duration
- resolution width/height
- FPS
- pipeline mode (classic, msr, or ingredients)

CLI:

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

Full-Auto stages exposed in Studio:

1. `Song brief`
2. `ACE-Step audio rendering`
3. `Project scaffold`
4. `Video pipeline`

The ACE-Step stage queues `workflows/audio_song_v2.json` in ComfyUI. FeverSlop patches the prompt/tags/lyrics/timing/seed inputs, waits for ComfyUI history completion, downloads the audio file, and writes it to:

```text
projects/<slug>/input/<slug>.mp3
```

Generated files include:

```text
projects/<slug>/
|-- config.json
|-- full_auto_song_spec.json
|-- lyrics.txt
`-- input/
    `-- <slug>.mp3
```

## MSR Mode

MSR mode maps to runner value:

```text
ltx_msr
```

MSR mode uses actor and location references instead of storyboard start frames.

Typical MSR outputs:

```text
projects/<slug>/output/references/
projects/<slug>/output/render/plans/references.json
projects/<slug>/output/render/scenes/
projects/<slug>/output/render/final/movie.mp4
```

MSR-specific Studio actions:

| Studio action | Backend action | Effect |
| --- | --- | --- |
| MSR references | `msr-references` | Render actor/location references. |
| MSR enrichment | `msr-enrich` | Rebuild MSR-enriched render plan/prompt fields. |

MSR mode relies on Scene Bible / Actor Bible style references. Actor references should generally be neutral-background character images; location references should remain real environment/background images.

## Ingredients Mode

Ingredients mode maps to runner value:

```text
ltx_ingredients
```

Ingredients mode composes a per-scene ingredients reference sheet from existing actor and location references, then renders using an ingredients-specific ComfyUI workflow that injects the song's audio as a latent.

Prerequisites:

- Actor and location references must exist (generated via MSR references step or the reference bible tool).
- The included song/music-video default is `workflows/video_ltxv_ingredients_audio_2stage_v6.json`; `--ingredients-workflow` may select another compatible workflow. The current workflow must expose the anchors required by its backend, plus an audio loader/trim branch when audio latent injection is enabled.

CLI:

```bash
uv run python run_pipeline.py ./projects/my-song \
  --video-pipeline ltx_ingredients \
  --skip-tests
```

Full-Auto with Ingredients mode:

```bash
uv run python full_auto.py \
  --idea "A cyberpunk chase through a futuristic city" \
  --style "dark synthwave with cinematic drums" \
  --project-name neon-wolves \
  --run-video-pipeline \
  --video-pipeline ltx_ingredients \
  --skip-tests
```

Studio:

- **Project Settings** -> set `video_pipeline` to `ltx_ingredients`.
- **Pipeline** -> **Full pipeline** to run the full ingredients pipeline.
- **Pipeline** -> **Ingredients sheets** to compose or refresh ingredients scene sheets.

Ingredients mode stages:

1. Main pipeline (audio, timeline, render plan).
2. Anchor fix.
3. Ingredients scene sheet composition - composes letterboxed reference sheets, creates stable global prompts, and writes compact `plans/ingredients.json` scenes.
4. LTX prepare/render - patches the temporal relay, scene sheet, and audio window into V4 workflows, then renders prepared scenes.
5. Final concat and audio mux.

Typical Ingredients mode outputs:

```text
projects/<slug>/output/references/ingredients_sheets/scene_0001_ingredients.png
projects/<slug>/output/references/ingredients_sheets/scene_0002_ingredients.png
projects/<slug>/output/render/plans/ingredients.json
projects/<slug>/output/render/scenes/scene_0001/workflow.json
projects/<slug>/output/render/scenes/scene_0001/final.mp4
projects/<slug>/output/render/final/movie.mp4
```

Ingredients-specific Studio actions:

| Studio action | Backend action | Effect |
| --- | --- | --- |
| Ingredients sheets | `ingredients-sheets` | Compose ingredients scene sheets from references. |

The ingredients workflow handles audio loading and trimming internally. The `MUX_ORIGINAL_AUDIO` stage remains active and muxes the full original song into the final output. `--ingredients-workflow` overrides the default workflow path.

## Classic Mode

Classic mode maps to runner value:

```text
ltx_i2v
```

Classic mode uses the original image-to-video path with storyboard/start-frame generation and does not require MSR Scene Bible / Actor Bible reference sheets.

Typical classic outputs:

```text
projects/<slug>/output/render/storyboard/
projects/<slug>/output/render/storyboard/index.html
projects/<slug>/output/render/ltx_single_prompt/
projects/<slug>/output/render/ltx_single_prompt/<project_name>.mp4
```

## Defaults

| Surface | Default |
| --- | --- |
| CLI `run_pipeline.py` | `--video-pipeline ltx_i2v` |
| CLI `full_auto.py` | `--video-pipeline ltx_i2v` unless passed through |
| Ingredients workflow default | `workflows/video_ltxv_ingredients_audio_2stage_v6.json` |
| Studio full-auto API payload | `pipeline_mode: "classic"` if omitted |
| Studio full-auto UI | sends `pipeline_mode: "msr"` by default |
| Studio Project Settings defaults | `video_pipeline: "ltx_msr"` |
| Full-Auto resolution | `1280 x 704` |
| Full-Auto FPS | `24` |
| Full-Auto allowed FPS | `16`, `24`, `50` |

## Studio Pipeline Actions

Standard project actions:

| UI label | Backend action |
| --- | --- |
| Main pipeline | `main-pipeline` |
| Relay compact | `relay-compact` |
| Anchor fix | `anchor-fix` |
| MSR references | `msr-references` |
| MSR reference sheets | `msr-reference-sheets` |
| MSR prompt enrichment | `msr-prompt-enrich` |
| Ingredients sheets | `ingredients-sheets` |
| Storyboard frames | `storyboard-frames` |
| Storyboard page | `storyboard-page` |
| Render selected scenes | `ltx-render-scenes` |
| Concat video only | `concat-video-only` |
| Mux original audio | `mux-original-audio` |
| Full pipeline | `full-pipeline` |

Full-Auto project action:

| UI label | Backend action |
| --- | --- |
| Full-auto pipeline | `full-auto` |

Movie project action:

| UI label | Backend action | Effect |
| --- | --- | --- |
| Movie references | `movie-references` | Renders actor/location reference sheets and fills `movie/references/manifest.json`. |
| Start movie production | `movie-full-auto` | Uses the scaffolded movie render plan and produces `output/movie/<slug>.mp4`. |

## Movie Pipeline

Movie projects are separate from standard music-video projects. Creation accepts
either:

- `short_story`: prose idea text
- `screenplay`: structured screenplay text with scene headings such as `INT.`
  or `EXT.`

Scaffold mode runs:

1. Story-Arch generation.
2. Movie Bible generation in `movie/bible.json`.
3. Story design in `movie/story_design.json`.
4. Canonical screenplay persistence in `movie/screenplay.json` and `movie/screenplay.md`.
5. Narrative memory planning in `movie/narrative_plan.json`.
6. Scene-card and shot-card persistence in `movie/scene_cards.json` and `movie/shot_cards.json`.
7. Movie continuity planning in `movie/continuity_plan.json`.
8. Scene/shot planning from the Bible and memory artifacts into `movie/render_plan.json`.
9. Reference manifest persistence from Bible actors/locations.

Full-auto mode starts the `movie-full-auto` job after scaffold creation. Studio
reports these job steps:

1. `Movie Bible`
2. `Movie Continuity`
3. `Render Plan`
4. `Movie references`
5. `Movie MSR prompt enrichment`
6. `LTX MSR native-audio render`
7. `Final movie`

Run `movie-references` to generate or refresh the actor and location MSR sheet
paths in `movie/references/manifest.json`. The manifest is generated from
`movie/bible.json`, not from shot text. `movie-full-auto` also runs this
reference step automatically when required paths are missing.

Movie stages are also available without Studio:

```powershell
uv run python movie_pipeline.py .\projects\tm3 --skip-movie-render
```

For MiniMax H3 Movie modes, `--skip-movie-render` still runs the reference and
H3 prompt preparation stages, then stops before ComfyUI video rendering. The
prepared scene-list prompt artifact is written to
`output/movie/<movie-video-workflow>/render_plan_h3.json` and is reused by the
subsequent render command:

```powershell
uv run python movie_pipeline.py .\projects\my-movie `
  --movie-video-workflow minimax-h3-r2v `
  --r2v-workflow .\workflows\video_minimax_h3_r2v_v1.json `
  --skip-movie-render

uv run python movie_pipeline.py .\projects\my-movie `
  --movie-video-workflow minimax-h3-r2v `
  --r2v-workflow .\workflows\video_minimax_h3_r2v_v1.json
```

This ensures `movie/bible.json`, `movie/story_design.json`, `movie/screenplay.json`,
`movie/narrative_plan.json`, `movie/scene_cards.json`,
`movie/shot_cards.json`, and `movie/continuity_plan.json`, writes
`movie/references/manifest.json` from the Bible, renders missing movie
reference sheets, and writes `movie/render_plan_msr.json`. It does not render
the movie.

Continue from existing references and render the movie:

```powershell
uv run python movie_pipeline.py .\projects\tm3 --skip-movie-references
```

Force reference regeneration, then stop before movie rendering:

```powershell
uv run python movie_pipeline.py .\projects\tm3 --force-movie-references --skip-movie-render
```

Prepare canonical per-shot Movie MSR or Ingredients workflows for inspection
without queueing ComfyUI. `--write-debug-workflows` is retained as a deprecated
compatibility alias for this prepare step:

```powershell
uv run python movie_pipeline.py .\projects\tm3 `
  --skip-movie-references `
  --write-debug-workflows `
  --scenes 1,3,5 `
  --skip-movie-render
```

This requires ready reference paths in `movie/references/manifest.json`. It
writes each selected shot's authoritative `workflow.json` and `manifest.json`
under `output/render/scenes/scene_####/`. `--debug-workflows-dir` is accepted
only for CLI compatibility; canonical scene paths are always used.

Development/test runs can use local placeholder backends:

```powershell
uv run python movie_pipeline.py .\projects\tm3 `
  --reference-backend local `
  --render-backend local
```

Movie skip flags:

| Flag | Behavior |
| --- | --- |
| `--skip-movie-bible` | Reuse existing `movie/bible.json`; fail if it is missing. |
| `--skip-movie-story-design` | Reuse existing `movie/story_design.json`; fail if it is missing. |
| `--force-movie-story-design` | Rebuild `movie/story_design.json` from project metadata/render plan. |
| `--skip-movie-screenplay` | Reuse existing `movie/screenplay.json`; fail if it is missing. |
| `--force-movie-screenplay` | Rebuild `movie/screenplay.json` from project metadata/render plan. |
| `--skip-movie-narrative` | Reuse existing `movie/narrative_plan.json`; fail if it is missing. |
| `--skip-movie-scene-cards` | Reuse existing `movie/scene_cards.json`; fail if it is missing. |
| `--skip-movie-shot-cards` | Reuse existing `movie/shot_cards.json`; fail if it is missing. |
| `--skip-movie-continuity` | Reuse existing `movie/continuity_plan.json`; fail if it is missing. |
| `--skip-movie-plan` | Reuse existing `movie/render_plan.json`; fail if it is missing. |
| `--skip-movie-references` | Reuse existing paths in `movie/references/manifest.json`. |
| `--skip-movie-msr-enrich` | Reuse existing `movie/render_plan_msr.json`; if missing, render the plain plan. |
| `--skip-movie-render` | Stop after Bible/plan/reference/MSR-enrichment stages. |
| `--force-movie-references` | Render references even when manifest paths already exist. |
| `--keyframe-mode none\|start\|start-end` | Add start/end frame prompt briefs to the MSR plan without changing video prompts. |
| `--movie-video-workflow msr\|msr-i2v-startframe\|i2v-edit\|startframe-director\|ingredients` | Select the movie workflow contract; MSR-only remains the default. |
| `--continuity-keyframes none\|last-to-start` | Experimental render-time chaining. With `last-to-start`, continuous movie shots use the previous rendered clip's last frame as the next `#STARTFRAME`. Default is off. |
| `--msr-i2v-workflow PATH` | Optional MSR-I2V workflow template used only for continuous shots with a generated startframe. `--msr-workflow` remains the hard-cut/base MSR workflow. |
| `--write-debug-workflows` | For MSR/Ingredients, deprecated alias that prepares canonical per-shot workflows without queueing ComfyUI; startframe-director retains its debug-export behavior. |
| `--debug-workflows-dir PATH` | For MSR/Ingredients, deprecated compatibility option because canonical `output/render/scenes/` paths are used; startframe-director still honors this directory. |
| `--reference-backend local\|comfyui` | Select reference generation backend. Defaults to ComfyUI. |
| `--render-backend local\|comfyui` | Select movie render backend. Defaults to ComfyUI. |

### Sequence-to-Sheet References

Actor and location references can optionally be generated from a short
multi-view sequence. Enable this with `--reference-generation sequence_sheet`
and provide a compatible I2VA workflow through
`--sequence-to-sheet-workflow`. The pipeline persists the anchor, sequence,
selected frames, contact sheet, and final tiled sheet so the intermediate
result can be reviewed before R2V/I2V rendering.

For reproducible project runs, store the choice in `config.json` instead of
repeating the CLI options:

```json
{
  "video_pipeline": "minimax-h3-r2v",
  "reference_generation": "sequence_sheet",
  "workflows": {
    "reference_sequence": "workflows/sequence_to_sheet_minimax_h3_i2va_v1.json"
  }
}
```

Explicit CLI options override the project configuration. Changing the
reference mode or configured sequence workflow invalidates incompatible
reference artifacts on the next safe resume.

The complete tutorial and copyable commands are in
[`documentation/sequence-reference-pipeline.md`](sequence-reference-pipeline.md).

### Scene Reference Sheet Composer

The scene reference sheet composer automatically stitches per-shot reference images
into a single letterboxed grid sheet. Unlike MSR sheets that crop to fill cells,
scene sheets preserve all content via letterboxing (contain mode) so that
heterogeneous images — portrait actor MSRs, landscape location heroes, and square
prop references — compose without losing any visual detail.

Typical outputs:

```text
movie/scene_sheets/shot_001_scene.png
movie/scene_sheets/shot_002_scene.png
...
```

Each sheet is written at the target render resolution (default `1280 x 704`) and
the path is attached to `movie/render_plan_msr.json` under
`shot["scene_reference_sheet"]`.

The composer is invoked automatically during `movie-full-auto` and when running
MSR prompt enrichment. It reads `movie/references/manifest.json`, resolves
actor/location sheet paths for each shot, and composes a grid with up to four
columns.

Programmatic use:

```python
from feverslop.application.movie_references import SceneReferenceSheetBuilder

builder = SceneReferenceSheetBuilder(
    project_dir="projects/my_movie",
    manifest=manifest,
    size=(1280, 704),
)
result = builder.build(shot)
# result["sheet_path"] -> "movie/scene_sheets/shot_001_scene.png"
# result["image_count"] -> 3
# result["images"] -> [{"path": ..., "type": "actor", "id": "actor_1"}, ...]
```

Low-level composition function:

```python
from feverslop.application.reference_bible import compose_scene_reference_sheet

compose_scene_reference_sheet(
    image_paths=[path1, path2, path3],
    output_path=Path("output/sheet.png"),
    size=(1280, 704),
    columns=3,
)
```

### Movie I2V/Edit Mode

`--movie-video-workflow i2v-edit` keeps the full Movie planning pipeline, then
creates a visual plan and classic I2V render plan instead of
`movie/render_plan_msr.json`.

Typical outputs:

```text
movie/visual_plan.json
movie/render_plan_i2v.json
output/movie/storyboard/index.html
output/movie/storyboard/base/
output/movie/storyboard/edit/
output/movie/storyboard/final/
output/movie/ltx_i2v/
output/movie/<slug>.mp4
```

This mode skips MSR video conditioning and MSR prompt enrichment. It may still
render or reuse character reference images because each edit pass needs one
actor identity input. Multi-character shots are composed iteratively with a
two-input edit workflow: current plate plus one character reference, one
character per pass.

Example:

```powershell
uv run python movie_pipeline.py .\projects\blackwood `
  --movie-video-workflow i2v-edit `
  --hero-workflow .\workflows\image_t2i_startframe_krea_v1.json `
  --edit-workflow .\workflows\image_edit_flux2_klein_2ref_v1.json `
  --i2v-workflow .\workflows\video_ltxv_i2v_v2.json
```

### Movie Startframe Director Mode

`--movie-video-workflow startframe-director` uses a multi-stage startframe
engine to produce high-quality first frames before LTX I2V handoff. Each shot
passes through these stages:

1. **Director frame** — Krea2 or Ideogram generates an initial layout image from
   the shot's startframe intent, actors, location, camera, and lighting data.
2. **Masking** — SAM3 segments each actor's region in the director frame.
3. **Identity repair** — SDXL with IP-Adapter inpaints each masked actor region
   using the character reference, preserving face, hair, wardrobe, and pose.
4. **Detail pass** — EasyUse/Impact Detailer refines the image for coherence.
5. **Validation** — Gemma4 checks identity, wardrobe, and action against the
   shot contract.
6. **LTX I2V** — the final startframe drives image-to-video with native audio.

The workflow reads `movie/startframe_plan.json`, `movie/startframe_director_prompts.json`,
and `movie/identity_ledger.json` to build per-shot prompts. Director prompts
use Krea2's natural-language format or Ideogram's structured schema depending
on `--startframe-director-backend`.

Typical outputs:

```text
output/movie/startframes/director/scene_0001.png
output/movie/startframes/masks/scene_0001_actor_id.png
output/movie/startframes/repair/scene_0001_actor_id.png
output/movie/startframes/detail/scene_0001.png
output/movie/storyboard/final/scene_0001.png
output/movie/startframe_validation.json
output/movie/ltx_startframe_director/
output/movie/startframe-director.mp4
```

CLI:

```powershell
uv run python movie_pipeline.py .\projects\my-movie `
  --movie-video-workflow startframe-director `
  --startframe-director-backend krea2 `
  --director-workflow .\workflows\image_t2i_startframe_krea_v1.json `
  --mask-workflow .\workflows\image_mask_sam3_actor_regions_v1.json `
  --identity-repair-workflow .\workflows\image_repair_sdxl_ipadapter_identity_v1.json `
  --detail-workflow .\workflows\image_detail_easyuse_startframe_v1.json `
  --i2v-workflow .\workflows\video_ltxv_i2v_native_audio_v2.json
```

Studio:

- **Project Settings** -> set `movie_video_workflow` to `startframe-director`.
- Set `movie_startframe_director_backend` to `krea2` or `ideogram`.
- Provide workflow paths or accept the defaults.

Required ComfyUI workflows:

| Workflow | Default path | Purpose |
| --- | --- | --- |
| Director | `workflows/image_t2i_startframe_krea_v1.json` or `image_t2i_startframe_ideogram_director_v1.json` | Initial layout |
| Mask | `workflows/image_mask_sam3_actor_regions_v1.json` | Actor segmentation |
| Repair | `workflows/image_repair_sdxl_ipadapter_identity_v1.json` | Identity inpaint |
| Detail | `workflows/image_detail_easyuse_startframe_v1.json` | Final refinement |
| I2V | `workflows/video_ltxv_i2v_native_audio_v2.json` | Video handoff |

### Movie Ingredients Mode

`--movie-video-workflow ingredients` renders the movie using the ingredients
video pipeline. Unlike MSR, which conditions on global reference sheets, the
ingredients pipeline uses per-shot target prompts composed from the render plan
and scene data.

This mode is lightweight: it skips MSR prompt enrichment, identity repair,
and the multi-stage director pipeline. It reads `movie/render_plan.json` and
uses the `ingredients_target_prompt` and `ingredients_scene_sheet_description`
fields from each shot's `ltx` block.

Typical outputs:

```text
output/movie/ltx_ingredients/
output/movie/<slug>.mp4
```

CLI:

```powershell
uv run python movie_pipeline.py .\projects\my-movie `
  --movie-video-workflow ingredients `
  --ingredients-workflow .\workflows\video_ltxv_ingredients_2stage_v6.json
```

Studio:

- **Project Settings** -> set `movie_video_workflow` to `ingredients`.

The ingredients workflow must anchor `#PROMPT_POSITIVE`, `#SEED`, `#WIDTH`,
`#HEIGHT`, `#FRAMES`, `#FRAMERATE`, and `#SAVE_VIDEO`.

### Movie MiniMax H3 Modes

The Movie pipeline also supports the MiniMax H3 video modes:

- `minimax-h3-r2v`: render from the actor/location reference images in each
  Movie render-plan scene; reference audio is optional.
- `minimax-h3-t2v`: render directly from the scene prompt without image
  references.
- `minimax-h3-i2v`: render from the scene prompt with optional
  `keyframes.startframe_path` and `keyframes.endframe_path` inputs. This is the
  intended extension point for future start/end-frame authoring.

All three modes use the existing MiniMax scene renderer and concatenate their
clips into `output/movie/<mode>/` and `output/movie/<project>.mp4`.

Example:

```powershell
uv run python movie_pipeline.py .\projects\my-movie `
  --movie-video-workflow minimax-h3-i2v `
  --i2v-workflow .\workflows\video_minimax_h3_t2v.json
```

The corresponding workflow must contain the MiniMax H3 node and the usual
`#PROMPT`, `#WIDTH`, `#HEIGHT`, `#FRAMES`, `#MEGAPIXELS`, `#SEED`, and
`#SAVE_VIDEO` anchors. R2V workflows additionally need the reference-image
anchors expected by `ComfyUIMiniMaxH3R2VBackend`; T2V/I2V workflows use the
`#T2V_START` and `#T2V_END` anchors when frame paths are present.

### Vision-enriched LTX prompts

Vision enrichment requires the configured OpenAI-compatible model to accept
vision chat content. Each actor image is attached individually in declared
order, followed by the location image; the composed Ingredients grid is not
sent to the model. Images are resized proportionally to a maximum width or
height of 1024 pixels before attachment.

Ingredients prompts keep two explicit sections: one
`### Reference Sheet Description` containing every bound reference id and its
visible facts, followed by one `### Target Description` containing a detailed
250-400 word, chronological, continuous full-frame shot direction. The prompt
forbids reproducing source framing, composition, borders, panels, or layout.
MSR uses a separate format: reference facts go only to the global prompt, while
each PromptRelay local prompt contains concise frame-local action, acting,
camera, and environment direction. Ingredients headings never enter MSR relay
inputs.

If vision is unavailable, an image is missing, transport fails, or the model
response fails strict validation, enrichment deterministically preserves the
existing metadata-derived prompt. Existing NAG and negative-prompt template
inputs are not changed by vision enrichment or workflow materialization.

### Movie Render Plan Artifacts

The movie render plan declares:

```json
{
  "audio_policy": "ltx_native",
  "visual_backends": ["krea2", "ltx_msr"]
}
```

`movie/continuity_plan.json` is the Movie continuity ledger. It contains a
style/character/location ledger, per-shot incoming/carryover/outgoing state,
and a narrative chain with cause/effect fields such as `cause_from_previous`
and `sets_up_next`.

`movie/story_design.json` is the dramaturgical authoring layer between the raw
idea and the canonical script. It tracks act structure, turning points,
setup/payoff threads, character arcs, and scene blueprints with purpose,
conflict, emotional turn, subtext, and dialogue function. `movie/screenplay.json`
is the deterministic story base after project creation. Downstream narrative
planning, scene cards, shot cards, and render plans should build from this
artifact instead of repeatedly reinterpreting the raw idea.
`movie/shot_cards.json` contains short current-shot briefs, including optional
start/end frame briefs for keyframe workflows.

`movie/render_plan_msr.json` is the render-time plan for LTX/MSR. It adds
minimal current-shot prompt-relay data. `msr_global_prompt` describes only the
reference images passed to the workflow: every actor reference in order, then
the final scene/background reference. `local_prompts` describes only the
current shot's timed action, camera, acting, and any exact diegetic dialogue in
natural prose. Audio policy, style lists, anti-music text, reference-sheet
instructions, and continuity contracts are not copied into the positive
prompt-relay text.

For movie projects, the MSR workflow must not provide a custom audio input.
`MovieWorkflowPatcher.strip_audio_inputs()` removes audio loader/trim nodes and
audio input links so LTX 2.3 can generate synchronized voice, effects, and
environmental audio natively. Non-movie MSR workflows still keep their normal
custom audio behavior.

Movie production uses this MSR template by default before patching:

```text
workflows/video_default_ltxv_msr_1actor_1background_v4.json
```

Experimental MSR-I2V startframe rendering is template-based, not node-composed.
If `--movie-video-workflow msr-i2v-startframe` is used, provide an MSR-I2V
workflow via `--msr-i2v-workflow` with anchors such as `#MSR_ACTOR_1`,
`#MSR_BACKGROUND`, `#MSR_LORA`, `#MSR_FRAME_COUNT`, `#PROMPT_RELAY`, and
`#STARTFRAME`. The patcher fills those anchors; it does not invent missing
nodes. `--msr-workflow` remains the base MSR-only workflow for scene 1 and hard
cuts.

`--continuity-keyframes last-to-start` is a separate, opt-in continuity aid for
non-hard cuts. It only applies to shots marked
`transition_from_previous: "continuous"` and conservatively skips transitions
that do not share location or actors with the previous shot. Before rendering a
continuous shot, the movie adapter extracts the previous rendered clip's last
frame to `output/movie/keyframes/scene_XXXX_to_YYYY_start.png` and patches that
image into `#STARTFRAME` for the target scene. Selective re-renders require the
previous clip to already exist; the adapter will not silently re-render
predecessor clips. Hard cuts and the default `none` mode do not use startframes
and stay on the base MSR workflow.

Continuous MSR-I2V shots use a short render-time handoff window instead of a
normal front preroll. The patched workflow starts with the previous shot's end
prompt for 18 frames, then switches to the current shot prompt. During that
window `#MSR_FRAME_COUNT` is set to 17 and `#MSR_GUIDE.frame_idx` is set to 18,
so the startframe/I2V path establishes the transition before MSR identity
guidance takes over. This handoff is written only into the patched ComfyUI
workflow/debug workflow, not into `movie/render_plan_msr.json`.

By default Studio uses `ComfyUIMovieVisualAdapter` for movie production. The
movie project metadata stores `render_backend: "comfyui"` and the MSR workflow
path used for patching. A local placeholder render backend exists only as an
explicit development/test option in the movie creation form or API payload.
The ComfyUI adapter renders each shot through the patched MSR workflow with
`upload_audio=False`, so no custom audio is supplied and LTX 2.3 owns voice,
effects, and environment audio.

The ComfyUI movie adapter reads `movie/references/manifest.json` and resolves
shot `reference_ids` to rendered actor/location MSR sheet paths. If those paths
are still empty, production rendering stops with a validation error instead of
queueing a broken workflow. Krea2/reference-sheet generation is therefore the
required upstream step before production movie rendering.

Movie projects also default to `reference_backend: "comfyui"` for reference
sheet generation. The default workflows stored in project metadata are:

```text
workflows/image_t2i_startframe_krea_v1.json
workflows/image_edit_flux2_klein_1ref_v1.json
```

Override them in the movie creation form's advanced execution section or by
passing `movie_hero_workflow` and `movie_edit_workflow` to the project creation
API.

## Progress Reporting

Studio jobs expose:

- job id
- project id and project type
- action and pipeline type
- status: `queued`, `running`, `succeeded`, `failed`
- overall progress
- current step
- per-step status
- elapsed runtime
- ETA field, currently `null` unless a future job can estimate it
- recent logs and full retained logs

Exact step percentages are only shown when a step has a real numeric progress value. Otherwise Studio shows an indeterminate running indicator. ACE-Step/ComfyUI audio rendering currently appears as the `ACE-Step audio rendering` step with indeterminate progress unless the underlying ComfyUI integration later exposes exact node progress.

## Log Streaming

Studio streams logs with Server-Sent Events:

```text
GET /api/jobs/<job-id>/logs
```

The Pipeline page opens this stream for the active job and appends lines to **Recent output**. The UI auto-scrolls only while the user is already at the bottom.

Log handling:

- Rich panels/tables/markup are rendered to readable plain text.
- stdout/stderr from CLI-style code is captured for Studio jobs.
- Logs are in memory and retained per process.
- Restarting the backend clears job history and logs.
- The right Jobs column shows per-job logs in a capped scroll area.

## Duplicate-Start Protection

Studio disables pipeline start buttons while any queued/running pipeline job exists for the project.

The backend also rejects duplicate starts for pipeline actions:

```text
Pipeline is already running for this project
```

This protects against duplicate starts from the Studio or other local callers. Buttons become available again after the job reaches a terminal status such as `succeeded` or `failed`.

## Failure Handling

If a job raises an exception:

- job status becomes `failed`
- current running step is marked failed
- the error text is exposed on the job object
- Pipeline page displays the error
- logs remain available while the backend process stays alive

External failures usually come from ComfyUI availability, missing workflow anchors, missing models/custom nodes, FFmpeg failures, LLM endpoint failures, or invalid project paths/config.
