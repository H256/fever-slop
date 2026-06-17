# LTX Prompt Anchor Fix

This patch fixes prompt drift where LTX shows the Z-Image startframe briefly and then cuts away to a tree, bark, rope, shadows, or macro detail.

## Files

- `ltx_prompt_anchor_fixer.py`
- `fix_ltx_prompt_anchors.py`
- `relay_direction_builder.py`
- `README_LTX_PROMPT_ANCHOR_FIX.md`

## What it does

For `vocals` and `mixed` scenes:

- LTX base prompt starts by preserving the exact startframe composition.
- The main subject must remain visible in the foreground.
- Environmental motion happens behind or around the subject.
- It forbids cutting away to only tree/bark/ropes/shadows/canopy/macro detail.

For singing relay segments:

- Main subject remains visible.
- Main subject sings/lip-syncs.
- Tree/bark/shadows/ropes are not allowed to sing.

## Recommended full execution after project reset

### 1. Generate render plan

```powershell
uv run python main.py `
  --project .\projects\my_frst_project\config.json `
  --app-config .\app_config.json `
  --concept-batch-size 10
```

### 2. Compact relay prompts

```powershell
uv run python compact_relay_prompts.py `
  --app-config .\app_config.json `
  --input-render-plan .\projects\my_frst_project\output\render\render_plan_comfyui_00056_.json `
  --output-render-plan .\projects\my_frst_project\output\render\render_plan_comfyui_00056__compact.json `
  --max-words 28
```

### 3. Anchor/fix LTX prompts

```powershell
uv run python fix_ltx_prompt_anchors.py `
  --input-render-plan .\projects\my_frst_project\output\render\render_plan_comfyui_00056__compact.json `
  --output-render-plan .\projects\my_frst_project\output\render\render_plan_comfyui_00056__compact_anchored.json
```

Optional custom subject:

```powershell
--subject-anchor "the old weary warrior man with weathered scarred face, salt-and-pepper beard, tattered leather armor, and heavy frayed cloak"
```

### 4. Render Z-Image storyboard

Use the anchored render plan so storyboard and LTX are aligned:

```powershell
uv run python render_storyboard.py `
  --app-config .\app_config.json `
  --render-plan .\projects\my_frst_project\output\render\render_plan_comfyui_00056__compact_anchored.json `
  --workflow .\workflows\zimage_api.json `
  --output-dir .\projects\my_frst_project\output\render\storyboard `
  --no-skip-existing
```

### 5. Test LTX Scene 16

```powershell
uv run python render_ltx.py `
  --app-config .\app_config.json `
  --render-plan .\projects\my_frst_project\output\render\render_plan_comfyui_00056__compact_anchored.json `
  --workflow .\workflows\autoprompt_relay_ltxv_i2v.json `
  --audio .\projects\my_frst_project\input\ComfyUI_00056_.mp3 `
  --storyboard-dir .\projects\my_frst_project\output\render\storyboard `
  --output-dir .\projects\my_frst_project\output\render\ltx `
  --scenes 16 `
  --no-skip-existing `
  --preroll-frames 6 `
  --tail-loss-frames 6 `
  --debug-workflows-dir .\projects\my_frst_project\output\render\ltx_debug
```

Check:

```text
.\projects\my_frst_project\output\render\ltx\final\scene_0016.mp4
```

### 6. Render all LTX scenes

```powershell
uv run python render_ltx.py `
  --app-config .\app_config.json `
  --render-plan .\projects\my_frst_project\output\render\render_plan_comfyui_00056__compact_anchored.json `
  --workflow .\workflows\autoprompt_relay_ltxv_i2v.json `
  --audio .\projects\my_frst_project\input\ComfyUI_00056_.mp3 `
  --storyboard-dir .\projects\my_frst_project\output\render\storyboard `
  --output-dir .\projects\my_frst_project\output\render\ltx `
  --no-skip-existing `
  --preroll-frames 6 `
  --tail-loss-frames 6 `
  --debug-workflows-dir .\projects\my_frst_project\output\render\ltx_debug
```

### 7. Concat

```powershell
ffmpeg -f concat -safe 0 `
  -i .\projects\my_frst_project\output\render\ltx\concat_list.txt `
  -c copy `
  .\projects\my_frst_project\output\render\ltx\final_video.mp4
```
