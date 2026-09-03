# MiniMax H3 Setup

## Prerequisites

- **ComfyUI** with MiniMax H3 custom nodes installed.
- **Models**: `minimax_h3_t2v` and `minimax_h3_r2v` model files in ComfyUI `models/` folder.
- **FFmpeg** on PATH for audio/video encoding and postprocessing.

## ComfyUI configuration

- Ensure `app_config.json` contains:
  ```json
  {
    "comfyui": {
      "base_url": "http://your-comfyui-server.local",
      "prompt_timeout_seconds": 1800
    }
  }
  ```
- Set `prompt_timeout_seconds` to 1800+ for multi-minute renders.
- Start ComfyUI with the appropriate `--listen` and `--port` arguments so it's reachable from FeverSlop.
- On AMD/ROCm machines, the H3 two-pass workflows ship with `#LATENT_UPSCALE`
  set to `device: "cuda"`. Set `comfyui.latent_upscaler_device` in
  `app_config.json` so the latent upscaler runs on the right device. The
  recommended value is `"auto"`: it asks the running ComfyUI server for its
  GPU and switches the node to `"rocm"` on AMD/Radeon GPUs (leaving the
  template value on any other GPU). You can also set an explicit `"rocm"` (or
  `"cpu"`) if you prefer to pin the device by hand:
  ```json
  {
    "comfyui": {
      "latent_upscaler_device": "auto"
    }
  }
  ```
  Leave it unset (or `null`) on NVIDIA machines; single-pass workflows are
  unaffected either way.

## Workflow files

Workflow JSONs are stored in `workflows/`:

- Video MiniMax H3 Text-To-Video (T2V): `workflows/video/minimax_h3/t2v.json`
- Video MiniMax H3 Reference-To-Video (R2V): `workflows/video/minimax_h3/r2v_audio_v1.json`

Place or reference them in `app_config.json` if the default locations differ.

## Project validation

Run the validation suite to confirm the backend is properly registered:

```
uv run python -m unittest tests.test_comfyui_minimax_h3_t2v_backend -v
uv run python -m unittest tests.test_comfyui_minimax_h3_r2v_backend -v
uv run python -m unittest tests.test_minimax_h3_integration -v
```

## MiniMax H3 model requirements

- Models must be accessible via `comfyui.base_url + "/object_info"`.
- If models are not available, the backend will fail during prompt building.

## Troubleshooting

- `ComfyUI unreachable` → Check `app_config.json` `comfyui.base_url` and network connectivity.
- `MiniMaxH3Text2VideoVideo not found` → Install MiniMax H3 custom nodes into ComfyUI custom_nodes/.
- Video/audio track missing → Ensure FFmpeg is installed and on PATH.
- Prompt timeout → Increase `comfyui.prompt_timeout_seconds` in `app_config.json`.
