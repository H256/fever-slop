# Music Video Pipeline Files

## Files

- `config.example.json`  
  Per-project config. Copy this into your project folder as `config.json`.

- `app_config.example.json`  
  Global config for LLM and later ComfyUI. Copy this into the working directory as `app_config.json`.

- `project_config.py`  
  Loads project-specific settings: input audio, video settings, audio models, scene settings, steering notes, overrides.

- `app_config.py`  
  Loads global settings: LLM endpoint/model and later ComfyUI endpoint.

- `main.py`  
  Rich-logged pipeline.

## Example structure

```text
projects/
└─ forest_song/
   ├─ config.json
   ├─ input/
   │  └─ comfyui_00056_.mp3
   └─ output/
```

## Run

```powershell
uv run python main.py --project ./projects/forest_song/config.json --app-config ./app_config.json
```

## Dependency

```powershell
uv pip install rich
```

Existing pipeline dependencies still apply:

```powershell
uv pip install openai librosa soundfile numpy openai-whisper
```

# Prompt Patch

This patch separates prompt generation into model-specific layers:

- `concept_prompts_*.json` — Stage 1 scene concepts.
- `scene_details_*.json` — camera motion + character motion.
- `scene_prompts_*.json` — both `zimage_prompt` and `ltx_base_prompt`.
- `render_plan_*.json` — final render plan.

Important:

- The resolved subject is injected into every scene prompt generation call.
- Z-Image prompts are still-image keyframe prompts.
- LTX prompts are video prompts.
- Prompt Relay stays frame-based.
- One scene = one cut = one LTX render pass.
- `frame_count = round(fps * duration_seconds) + 1`.

Add these project config fields if missing:

```json
"trigger_word": "",
"steering": {
  "zimage": "Create strong still-image keyframes. Avoid motion language, lip sync, singing, and actions that require multiple frames.",
  "ltx": "Create video-ready prompts with controlled motion and strong continuity. Preserve subject identity, wardrobe, lighting, and environment."
}
```
