# `app_config.json` reference

`app_config.json` configures the external services and optional workflow
profiles used by the CLI. Copy
[`app_config.example.json`](../app_config.example.json) as a starting point;
the file itself is local and should not contain credentials in committed
changes.

All fields are optional unless stated otherwise. If a section or field is
omitted, the default shown below is used.

## Complete top-level structure

```json
{
  "llm": {},
  "comfyui": {},
  "execution": {},
  "video_workflow_profiles": [],
  "storyboard_prompt_transforms": []
}
```

## `execution`

`execution.vram_handoff` controls machine-local resource transitions for the
safe music-video resume command. It does not belong in a project config.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `vram_handoff` | `continuous` or `manual` | `continuous` | Whether safe resume may cross directly between LLM- and ComfyUI-owned stages. |

Use `manual` when one GPU cannot keep the external LLM and ComfyUI models
loaded together:

```json
{
  "execution": {
    "vram_handoff": "manual"
  }
}
```

One invocation then executes at most one contiguous LLM or ComfyUI phase.
Workflow preparation belongs to the ComfyUI phase because it contacts the live
backend and uploads assets. Neutral work such as plan synchronization,
reference binding, and muxing stays attached to the surrounding phase. At the
next ownership change the command exits successfully and tells the operator
what to unload/load. After the model swap, repeat the unchanged command:

```bash
uv run python main.py run PROJECT --resume
```

No cursor file is needed: the next invocation derives its position from the
canonical artifacts and scene checkpoints. MSR-derived plans record
`stage_provenance.msr_prompt_enrich.input_fingerprint`, so a completed
reference phase cannot be mistaken for completed prompt enrichment. Explicit
compatibility `--stage`
commands are intentionally not partitioned; their resource lifecycle remains
the operator's responsibility.

## `llm`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `api_key` | string | none | Optional API key. Prefer `LLM_API_KEY` in the process environment or adjacent `.env` for secrets. |
| `base_url` | string | `http://localhost:8080/v1` | OpenAI-compatible API base URL. |
| `model` | string | `default` | Model name sent to the endpoint. |
| `models` | object | `{}` | Optional task-profile model overrides, for example `{ "creative": "story-model", "structured": "fast-model" }`. Missing profiles fall back to `model`. |
| `temperature` | number | `0.7` | Sampling temperature for LLM requests. |
| `dspy_temperature` | number | `0.4` | Sampling temperature for DSPy H3 planner, analyzer, and renderer calls. |
| `max_tokens` | integer | `4096` | Maximum completion token count. |
| `request_timeout_seconds` | number | `180.0` | Timeout for an LLM request. |
| `dspy_cache` | boolean | `false` | Whether DSPy may reuse cached LM responses. Set to `true` only when that behavior is wanted. |
| `max_concurrent_requests` | integer | `1` | Process-local ceiling shared by direct OpenAI-compatible calls and DSPy/LiteLLM calls. Different values in one Python process are rejected so the shared budget stays explicit. |
| `prompt_judge_attempts` | integer | `3` | Maximum number of final-prompt composer attempts after DSPy judge feedback. After the last bad result, the prompt and judge history are saved and rendering continues. |

The API-key precedence is: process environment, `llm.api_key`, then
`LLM_API_KEY` from the `.env` file next to `app_config.json`.

`llm.max_concurrent_requests` coordinates only threads in the current
FeverSlop Python process. It does not coordinate multiple FeverSlop processes,
other clients, or slots already accepted by the LLM server. Use server-side or
deployment-level limits if several processes can call the same model endpoint.

The legacy single `llm.model` setting remains sufficient. Profile overrides
are optional and are only selected by pipeline code that explicitly requests a
profile; they do not change the Thinking/No-Thinking server configuration.

DSPy prompt modules use conservative per-signature completion budgets. Short
structured prompt results are limited to 150 words and use a smaller token
budget; creative signatures receive a larger budget. These are output
contracts and do not enable or disable server-side Thinking.

## `comfyui`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `base_url` | string | `http://127.0.0.1:8188` | ComfyUI HTTP API base URL. |
| `prompt_timeout_seconds` | number | `1800.0` | Maximum time to wait for a queued prompt. |
| `model_overrides` | array | `[]` | Optional strict model replacement rules; see the fields below. |
| `default_max_render_duration_seconds` | number or `null` | `null` | Fallback maximum render duration when no workflow-specific limit applies. |
| `video_workflow_limits` | array | `[]` | Per-workflow maximum render durations; see the fields below. |
| `latent_upscaler_device` | string or `null` | `null` | ComfyUI device for the `#LATENT_UPSCALE` node in MiniMax H3 two-pass video workflows. One of `"cuda"`, `"rocm"`, or `"cpu"`. `null` keeps the template default (`"cuda"`), which is correct on NVIDIA machines. |

Each `model_overrides` item has all of these string fields:

```json
{
  "workflow": "workflows/example.json",
  "node_id": "42",
  "node_title": "CheckpointLoaderSimple",
  "input": "ckpt_name",
  "expected_value": "old-model.safetensors",
  "replacement": "new-model.safetensors"
}
```

Each `video_workflow_limits` item has:

```json
{
  "workflow": "workflows/video_example.json",
  "max_render_duration_seconds": 18.0
}
```

Workflow paths in these settings are repository-relative. Durations must be
finite and greater than zero. Duplicate workflow limits are rejected.

Set `comfyui.latent_upscaler_device` only when ComfyUI runs the MiniMax H3
two-pass latent upscaler on a non-CUDA device. For example, on an AMD/ROCm
machine:

```json
{
  "comfyui": {
    "latent_upscaler_device": "rocm"
  }
}
```

The value is applied to the `#LATENT_UPSCALE` node at queue time and has no
effect on single-pass workflows, which do not contain that node.

## `video_workflow_profiles`

Profiles select alternative workflows for a pipeline and purpose. A profile
must contain `name`, `pipeline`, `workflow`, `purpose`, `stages`,
`output_scale`, and `supports_per_pass_loras`.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | string | required | Unique profile name used for selection. |
| `pipeline` | string | required | Pipeline identifier, such as `ltx_ingredients`. |
| `workflow` | string | required | Repository-relative workflow path. |
| `purpose` | `preview` or `final` | required | Whether the profile is for preview or final output. |
| `stages` | `1` or `2` | required | Number of workflow stages. |
| `output_scale` | positive number | required | Output scale handled by the workflow. |
| `supports_per_pass_loras` | boolean | required | Whether per-pass LoRA settings are supported. |
| `supports_start_frame` | boolean | `false` | Whether the workflow accepts a start frame. |
| `satisfies_final_output` | boolean or `null` | inferred from `purpose` | Explicitly declares whether the profile can produce final output. |
| `default` | boolean | `false` | Makes this profile the default for its pipeline/purpose pair. Only one default is allowed per pair. |

Example:

```json
{
  "name": "ingredients-final",
  "pipeline": "ltx_ingredients",
  "workflow": "workflows/video_ltxv_ingredients_audio_2stage_v6.json",
  "purpose": "final",
  "stages": 2,
  "output_scale": 1.0,
  "supports_per_pass_loras": true,
  "supports_start_frame": false,
  "satisfies_final_output": true,
  "default": true
}
```

## `storyboard_prompt_transforms`

Each item defines a prompt transformation workflow. `workflow` is required;
the other fields have defaults.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `workflow` | string | required | Workflow used for the transformation. |
| `kind` | string | `template` | Transformation kind. |
| `template` | string | `""` | Optional prompt template path or value. |
| `positive_prompt_input` | string | `text` | Workflow input receiving the positive prompt. |
| `debug_dir` | string | `storyboard_prompt_debug` | Directory for transformation debug artifacts. |

Example:

```json
{
  "workflow": "workflows/image/image-model/image_t2i_startframe_ideogram_v1.json",
  "kind": "template",
  "template": "documentation/ideogram4_prompt_template.md",
  "positive_prompt_input": "text",
  "debug_dir": "ideogram4_prompt_debug"
}
```

After changing service URLs, workflow paths, model overrides, or profiles,
validate the affected ComfyUI workflows before running a render. See
[`documentation/setup.md`](setup.md) for the validation command and service setup.
