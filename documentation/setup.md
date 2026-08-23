# Setup

This document covers local setup for the FeverSlop CLI.

## Prerequisites

Required:

- Python `3.12`.
- `uv` for Python dependency management.
- FFmpeg available in `PATH`.
- ComfyUI running with the API enabled.
- Required ComfyUI workflows and models under `workflows/` and the configured ComfyUI model folders.
- An OpenAI-compatible LLM endpoint.

Full-Auto also requires an ACE-Step ComfyUI workflow:

```text
workflows/audio_song_v2.json
```

## Install Dependencies

From the repository root:

```bash
uv sync
```

## Python Dependencies

Python dependencies are declared in `pyproject.toml`. Important runtime dependencies include:

- `rich` for CLI output.
- `openai` for OpenAI-compatible LLM calls.
- `demucs`, `openai-whisper`, `librosa`, `soundfile`, and PyTorch packages for audio analysis.
- `pillow` for image processing.

The project uses a PyTorch CUDA index configured in `pyproject.toml`.

## Environment Variables

Runtime services are configured through `app_config.json`. The LLM API key can
also be stored in a local `.env` file next to `app_config.json`.

Both `app_config.json` and `.env` are ignored by Git. Do not add credentials to
`app_config.example.json` or commit another machine-specific config file.

## `app_config.json`

Runtime infrastructure is configured by `app_config.json` in the repository
root. Use `app_config.example.json` as a template. The complete list of
supported settings, nested fields, and defaults is maintained in the
[app_config reference](app_config.md).

Instead of storing the key in JSON, create `.env` next to `app_config.json`:

```dotenv
LLM_API_KEY=your-local-key
```

Key resolution order is:

1. `LLM_API_KEY` already set in the process environment
2. `llm.api_key` in `app_config.json`
3. `LLM_API_KEY` in the adjacent `.env`

Restart FeverSlop after editing either local file. `.env` is read directly and
does not modify the process environment.

## ComfyUI Requirements

ComfyUI must be reachable at `comfyui.base_url`.

### API endpoint safety

ComfyUI and OpenAI-compatible LLM URLs must use `http` or `https` and may not
contain embedded credentials, query strings, or fragments.

Private LAN addresses are allowed by default because local-network services
are the normal deployment model, and loopback addresses remain valid for
local services. Deployments that need strict SSRF filtering can pass
`allow_private_addresses=False` to the API client constructors. In strict
mode, literal private, link-local, multicast, reserved, and unspecified IP
addresses are rejected, and other hostnames are resolved via DNS: when DNS
returns addresses, every resolved address is checked the same way, while
hostnames that do not resolve are left to the HTTP client. Loopback stays
allowed in strict mode for local services.

To restrict outbound API hosts further, set a comma-separated allowlist
(case-insensitive):

```text
FEVERSLOP_ALLOWED_API_HOSTS=render-box,llm.example
```

The allowlist applies regardless of the address-filtering mode: hosts that
are not listed are rejected, and an allowlisted host is explicitly trusted
and skips the address checks, even in strict mode. That keeps narrowly
trusted endpoints available in either mode.

API clients can add authentication through their configured API keys or
headers. They also accept a configurable `min_request_interval_seconds`
value; the default is `0` for local development. This is a per-process
client limiter, not a network-wide abuse-prevention mechanism.

Requests do not follow redirects, so a redirect cannot bypass the configured
API host policy.

The selected workflows must exist and use the node anchors expected by the pipeline. Common workflow files:

```text
workflows/audio_song_v2.json
workflows/image_t2i_startframe_v1.json
workflows/image_t2i_startframe_krea_v1.json
workflows/image_edit_flux2_klein_1ref_v1.json
workflows/video_ltxv_i2v_v2.json
workflows/video_ltxv_msr_1actor_1background_v4.json
workflows/video_seedvr2_3b_api.json
```

For SeedVR2, the ComfyUI server must expose `ResizeImageMaskNode`,
`GetVideoComponents`, `VAEEncodeTiled`, `VAEDecodeTiled`, `CreateVideo`, and
`SaveVideo`, as well as `SeedVR2Preprocess`, `SeedVR2Conditioning`,
`SeedVR2TemporalChunk`, `SeedVR2TemporalMerge`, and `SeedVR2PostProcessing`.
The workflow also uses `ComfySwitchNode` and `PrimitiveBoolean` for the
reference-template's temporal latent split/merge path, plus
`UNETLoader` and `VAELoader` for the configured model and VAE. Validate the workflow against the active server
before rendering:

```bash
uv run python -m feverslop.tools.validate_comfyui_workflows \
  --app-config ./app_config.json \
  --workflows-dir ./workflows
```

Full-Auto ACE-Step workflow contract for `workflows/audio_song_v2.json`:

| Node title | Patched inputs |
| --- | --- |
| `ACE_STEP` | `tags`, `lyrics`, `bpm`, `duration`, `language`, `keyscale`, `timesignature`, `seed` |
| `KSampler` | `seed` |
| `Empty Ace Step 1.5 Latent Audio` | `seconds` |
| `SAVE` | `filename_prefix` |

Model references are resolved against the configured ComfyUI server before queueing workflows. See [ComfyUI model resolution](comfyui_model_resolution.md).

## Validate Workflows

After changing ComfyUI servers, workflows, models, or overrides:

```bash
uv run python -m feverslop.tools.validate_comfyui_workflows \
  --app-config ./app_config.json \
  --workflows-dir ./workflows
```

This validates model references against the configured ComfyUI server. It does not modify workflow files.

## Required Directories

Projects are discovered under:

```text
projects/
```

Each project should be a direct child directory:

```text
projects/my-song/
```

Some legacy application components still use the following internal metadata
and thumbnail-cache directory under a project:

```text
projects/my-song/.studio/
```

The former Studio application is deprecated and is not a supported user
interface. The `.studio/` directory is retained only for compatibility with
those remaining components and is excluded from normal project artifact lists.
