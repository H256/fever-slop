# Autoprompter Music Video Pipeline

This patch fixes scene duration handling and LTX relay rendering.

## Key rule

Project config controls scene duration:

```json
"scene_generation": {
  "min_duration": 2.0,
  "max_duration": 10.0,
  "bias": 0.7,
  "duration_preset": "impact_weighted",
  "seed": 42
}
```

All beat scenes should respect this range before Stage1, storyboard, and LTX rendering.

If a generated scene is shorter than `min_duration`, it is merged.
If a generated scene is longer than `max_duration`, it is split.

This is applied as early as possible: directly after beat SRT generation.

---

## Files included

- `scene_duration_enforcer.py`
- `repair_scene_srt.py`
- `main_scene_duration_patch_snippet.py`
- `render_plan_normalizer.py`
- `normalize_render_plan.py`
- `ltx_video_renderer.py`
- `render_ltx.py`
- `README_COMPLETE_AUTOPROMPTER_PIPELINE.md`

---

## Patch main.py

Add import:

```python
from scene_duration_enforcer import (
    enforce_scene_srt_file,
    parse_scene_srt,
    validate_scene_durations,
)
```

Use two SRT paths:

```python
scene_srt_raw = timeline_dir / f"scenes_{song_id}_raw.srt"
scene_srt = timeline_dir / f"scenes_{song_id}.srt"
```

After beat generation, write raw SRT first:

```python
scene_generator.generate_from_json_file(
    beat_json_path=beat_json,
    output_srt_path=scene_srt_raw,
)
```

Then repair:

```python
enforce_scene_srt_file(
    input_srt=scene_srt_raw,
    output_srt=scene_srt,
    min_duration=scene_cfg.min_duration,
    max_duration=scene_cfg.max_duration,
)
```

Then validate:

```python
duration_errors = validate_scene_durations(
    parse_scene_srt(scene_srt),
    min_duration=scene_cfg.min_duration,
    max_duration=scene_cfg.max_duration,
)

if duration_errors:
    raise ValueError(
        "Scene duration constraints failed after repair:\n"
        + "\n".join(duration_errors)
    )
```

After this point, all downstream steps use the repaired `scene_srt`.

---

## Full generation

```powershell
uv run python main.py `
  --project .\projects\my_frst_project\config.json `
  --app-config .\app_config.json
```

Output render plan:

```text
.\projects\my_frst_project\output\render\render_plan_comfyui_00056_.json
```

---

## Optional safety repair for an existing render plan

If you already have a render plan with bad durations:

```powershell
uv run python normalize_render_plan.py `
  --input-render-plan .\projects\my_frst_project\output\render\render_plan_comfyui_00056_.json `
  --output-render-plan .\projects\my_frst_project\output\render\render_plan_comfyui_00056__duration_fixed.json `
  --min-duration 2.0 `
  --max-duration 10.0
```

Prefer regenerating from repaired SRT when possible. This normalizer is only a safety net.

---

## Compact relay prompts

If you use the previous relay compaction patch:

```powershell
uv run python compact_relay_prompts.py `
  --app-config .\app_config.json `
  --input-render-plan .\projects\my_frst_project\output\render\render_plan_comfyui_00056__duration_fixed.json `
  --output-render-plan .\projects\my_frst_project\output\render\render_plan_comfyui_00056__duration_fixed_compact.json
```

---

## Render storyboard

Use the final render plan that has valid scene durations:

```powershell
uv run python render_storyboard.py `
  --app-config .\app_config.json `
  --render-plan .\projects\my_frst_project\output\render\render_plan_comfyui_00056__duration_fixed_compact.json `
  --workflow .\workflows\zimage_api.json `
  --output-dir .\projects\my_frst_project\output\render\storyboard
```

If you regenerated main.py with repaired SRT directly, use the normal `render_plan_comfyui_00056_.json`.

---

## Render LTX

```powershell
uv run python render_ltx.py `
  --app-config .\app_config.json `
  --render-plan .\projects\my_frst_project\output\render\render_plan_comfyui_00056__duration_fixed_compact.json `
  --workflow .\workflows\autoprompt_relay_ltxv_i2v.json `
  --audio .\projects\my_frst_project\input\ComfyUI_00056_.mp3 `
  --storyboard-dir .\projects\my_frst_project\output\render\storyboard `
  --output-dir .\projects\my_frst_project\output\render\ltx `
  --min-duration 2.0 `
  --max-duration 10.0 `
  --debug-workflows-dir .\projects\my_frst_project\output\render\ltx_debug_workflows
```

---

## PromptRelay frame rule

Your workflow showed this pattern:

```text
#FRAMES = 5
#PROMPT_RELAY.segment_lengths = 4
```

Therefore default is:

```text
sum(segment_lengths) = frame_count - 1
```

The renderer uses this by default:

```powershell
--segment-length-mode frames_minus_one
```

If your node expects `sum(segment_lengths) = frame_count`, use:

```powershell
--segment-length-mode frames
```

---

## Debug checklist

When a generated video is 0 seconds, inspect the debug workflow:

```text
#FRAMES
#FRAMERATE
#TRIM_AUDIO.start_index
#TRIM_AUDIO.duration
#PROMPT_RELAY.segment_lengths
#SAVE_VIDEO.trim_to_audio
```

Expected with 24 fps and 2 seconds minimum:

```text
#FRAMES >= 49
#TRIM_AUDIO.duration >= 2.0 approx
sum(segment_lengths) = #FRAMES - 1
```

---

## Concat

After LTX rendering:

```powershell
ffmpeg -f concat -safe 0 -i .\projects\my_frst_project\output\render\ltx\concat_list.txt -c copy .\projects\my_frst_project\output\render\ltx\final_concat.mp4
```

# Concept Prompt Batch Mode

This patch adds optional batched generation for `concept_prompts.json`.

## Why?

Large single-call JSON generation can drop keys near the end, for example:

```text
stage1_segments has segment_040
concept_prompts only contains segment_001..segment_039
```

Batch mode reduces that risk by generating concepts in smaller chunks.

## Does continuity survive batching?

Yes, because every batch receives:

- `story_idea`
- `global_context`
- steering notes
- previous generated concepts
- a rolling summary of the visual story so far

So the LLM still knows where the story came from and where the next batch should continue.

## Files

- `concept_prompt_batcher.py`
- `main.py` patched with `--concept-batch-size`
- `prompt_pipeline_batch_patch.py` as manual patch reference
- `README_CONCEPT_BATCH_MODE.md`

## Usage

Recommended:

```powershell
uv run python main.py `
  --project .\projects\my_frst_project\config.json `
  --app-config .\app_config.json `
  --concept-batch-size 10
```

Disable batch mode:

```powershell
--concept-batch-size 0
```

Use smaller batches if the model is weak at JSON:

```powershell
--concept-batch-size 5
```

## Where this runs

This only affects concept generation:

```text
stage1_segments.json
→ concept_prompts.json
```

Everything after that stays the same:

```text
concept_prompts.json
→ scene_details.json
→ scene_prompts.json
→ render_plan.json
```

## Validation

The patched `main.py` validates:

- every `stage1_segments[].segment_id` must exist in `concept_prompts`
- extra keys are removed
- output order follows `stage1_segments`

If a key is missing even after batch repair, `main.py` raises a clear error.

# LTX Rolling Frames Patch

Adds rolling-frame rendering to reduce visible gaps between separately rendered LTX clips.

## Files

- `ltx_video_renderer.py`
- `render_ltx.py`
- `video_postprocessor.py`
- `trim_existing_ltx_clips.py`
- `README_ROLLING_FRAMES_LTX.md`

## Output structure

```text
output/render/ltx/
├─ raw/
│  ├─ scene_0001_raw.mp4
│  └─ ...
├─ final/
│  ├─ scene_0001.mp4
│  └─ ...
├─ concat_list.txt
└─ render_manifest.json
```

`concat_list.txt` points to `final/`, not `raw/`.

## Recommended command

Use your compact render plan if available.

```powershell
uv run python render_ltx.py `
  --app-config .\app_config.json `
  --render-plan .\projects\my_frst_project\output\render\render_plan_comfyui_00056__compact.json `
  --workflow .\workflows\autoprompt_relay_ltxv_i2v.json `
  --audio .\projects\my_frst_project\input\ComfyUI_00056_.mp3 `
  --storyboard-dir .\projects\my_frst_project\output\render\storyboard `
  --output-dir .\projects\my_frst_project\output\render\ltx `
  --preroll-frames 6 `
  --tail-loss-frames 6 `
  --debug-workflows-dir .\projects\my_frst_project\output\render\ltx_debug
```

Test scene 16 only:

```powershell
--scenes 16
```

## Tuning

Start with:

```powershell
--preroll-frames 6 --tail-loss-frames 6
```

If gaps remain:

```powershell
--preroll-frames 8 --tail-loss-frames 8
```

If starts get too unstable:

```powershell
--preroll-frames 4 --tail-loss-frames 6
```

## Concat

```powershell
ffmpeg -f concat -safe 0 `
  -i .\projects\my_frst_project\output\render\ltx\concat_list.txt `
  -c copy `
  .\projects\my_frst_project\output\render\ltx\final_concat.mp4
```

If streamcopy concat fails, re-encode:

```powershell
ffmpeg -f concat -safe 0 `
  -i .\projects\my_frst_project\output\render\ltx\concat_list.txt `
  -c:v libx264 -crf 18 -preset slow `
  -c:a aac -b:a 192k `
  .\projects\my_frst_project\output\render\ltx\final_concat_reencoded.mp4
```

## Note about prompt drift

Rolling frames reduces seam gaps. It does not fix prompt drift where:

```text
Z-Image: character foreground
LTX base prompt: forest / tree / gallows without character foreground
```

For that, make sure to run the compact relay prompt step and later tighten the LTX base prompt builder so it preserves the startframe composition.
