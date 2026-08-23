# ComfyUI Workflow Model Resolution

This project keeps ComfyUI workflow JSON files as portable templates. Before a workflow is sent to ComfyUI, FeverSlop resolves model references against the ComfyUI server configured for the current run.

This is primarily for switching between machines, for example:

- local Windows ComfyUI at `http://127.0.0.1:8188`
- remote Linux ComfyUI at `http://render-box:8188`

The workflow files are not rewritten. Resolution happens in memory before each render request is queued.

## Configuration

`app_config.json` selects the ComfyUI server:

```json
{
  "comfyui": {
    "base_url": "http://127.0.0.1:8188",
    "model_overrides": []
  }
}
```

`app_config.json` is local machine configuration and should not be committed. Use `app_config.example.json` as the shared template.

When `comfyui.base_url` changes, the next CLI run uses that server and resolves workflow model values against that server's dropdown values.

## Runtime Behavior

For every rendered workflow:

1. FeverSlop loads the workflow JSON from `workflows/`.
2. The normal dynamic patches are applied, such as prompts, seeds, frame counts, audio, start frames, and LoRA settings.
3. The resolver requests `GET /object_info` from the configured ComfyUI server.
4. For each workflow node, the resolver inspects only string inputs that ComfyUI exposes as dropdown values for that node class.
5. The resolver replaces the workflow value with the exact dropdown value expected by the current server.
6. The patched workflow is sent to `/prompt`.

Because `/object_info` is queried from the configured server, the same workflow can be used against different servers on separate runs.

Example:

```text
Workflow value:  LTXV2\ltx23-kwh-balanced-av.comfy.safetensors
Linux dropdown:  LTXV2/ltx23-kwh-balanced-av.comfy.safetensors
Queued value:    LTXV2/ltx23-kwh-balanced-av.comfy.safetensors
```

On a later run, if `app_config.json` points back to localhost, the resolver queries localhost and resolves against localhost's dropdown values instead.

## Matching Rules

The resolver tries matches in this order:

1. Exact dropdown value.
2. Slash-normalized value, so `\` and `/` path separators are treated as equivalent.
3. Basename match, only if exactly one dropdown value has that basename.

The resolver fails before rendering when a model cannot be resolved.

Missing model example:

```text
ComfyUI model reference 'foo.safetensors' for LoraLoader.lora_name in workflows/x.json node 12 was not found in server dropdown values.
```

Ambiguous basename example:

```text
Ambiguous ComfyUI model reference 'foo.safetensors' for LoraLoader.lora_name in workflows/x.json node 12. Matches: a/foo.safetensors, b/foo.safetensors
```

This is intentional. Rendering with the wrong model is worse than failing early.

## Manual Overrides

Most cases should be handled automatically. Use `model_overrides` only when a workflow or custom node cannot be resolved safely from dropdown values alone.

Override format:

```json
{
  "comfyui": {
    "base_url": "http://127.0.0.1:8188",
    "model_overrides": [
      {
        "workflow": "workflows/video_ltxv_i2v_v2.json",
        "node_id": "99",
        "node_title": "#LORA_1",
        "input": "lora_name",
        "expected_value": "LTXV2\\ltx23-kwh-balanced-av.comfy.safetensors",
        "replacement": "LTXV2/ltx23-kwh-balanced-av.comfy.safetensors"
      }
    ]
  }
}
```

Overrides are strict. An override is considered stale and fails if:

- the workflow path no longer matches
- the node id no longer exists
- the node title changed
- the input no longer exists
- the current workflow value differs from `expected_value`

This protects against workflow updates silently applying an old override to the wrong node or value.

## Validation Command

Validate all workflow templates against the currently configured ComfyUI server:

```powershell
.\.venv\Scripts\python.exe -m feverslop.tools.validate_comfyui_workflows `
  --app-config .\app_config.json `
  --workflows-dir .\workflows
```

The command reports how many model references would be patched. It does not modify workflow files.

Use this after:

- changing `comfyui.base_url`
- moving renders to another operating system
- updating ComfyUI models
- exporting or replacing workflow JSON files
- changing `model_overrides`

## Debug Workflows

For LTX renders, `--debug-workflows-dir` writes the final patched workflow JSON used for a scene. This includes dynamic prompt/audio/frame patches and model resolution results.

Example:

```powershell
.\.venv\Scripts\python.exe render_ltx.py `
  --app-config .\app_config.json `
  --render-plan .\projects\my_song\output\render\render_plan.json `
  --workflow .\workflows\video_ltxv_i2v_v2.json `
  --render-mode single_prompt `
  --audio .\projects\my_song\input\song.mp3 `
  --storyboard-dir .\projects\my_song\output\render\storyboard `
  --output-dir .\projects\my_song\output\render\ltx `
  --scenes 1 `
  --debug-workflows-dir .\projects\my_song\output\render\ltx_debug
```

Inspect the saved workflow if ComfyUI rejects a prompt or if a model value looks suspicious.

## Limitations

- Resolution is per process. Normal CLI runs start a fresh process, so they query the configured server each run.
- A future long-running service should either create a fresh resolver per render or explicitly refresh `/object_info`.
- The resolver only patches ComfyUI dropdown string inputs. It intentionally ignores connection arrays and non-string values.
- The resolver does not install, download, or copy models. The selected ComfyUI server must already have the required models.
