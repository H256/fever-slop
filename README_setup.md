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
