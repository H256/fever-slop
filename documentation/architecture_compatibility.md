# Architecture Compatibility

The public command-line entry points remain at the repository root for backwards compatibility:

- `main.py --project --app-config --concept-batch-size`
- `render_storyboard.py --render-plan --workflow --output-dir`
- `render_ltx.py --render-plan --workflow --render-mode --audio --storyboard-dir --output-dir`
- `compact_relay_prompts.py --input-render-plan --output-render-plan`
- `fix_ltx_prompt_anchors.py --input-render-plan --output-render-plan`
- `storyboard_page.py --render-plan --storyboard-dir --output-html`
- `normalize_render_plan.py --input-render-plan --output-render-plan`
- `repair_scene_srt.py --input-srt --output-srt`
- `trim_existing_ltx_clips.py --render-plan --raw-dir --output-dir`

Compatibility facades remain available for older imports:

- `ltx_video_renderer.LTXVideoRenderer`; new code should import from `feverslop.adapters.comfyui_video_backend`.
- `storyboard_renderer.StoryboardRenderer`; new code should import from `feverslop.adapters.storyboard_renderer`.
- `workflow_patcher.WorkflowPatcher`; new code should import from `feverslop.adapters.workflow_patcher`.

Current architecture boundaries:

- `feverslop.application` contains use cases and pipeline services; the application layer does not import concrete adapters.
- `feverslop.composition` wires configs, use cases, and concrete adapters for CLI entry points.
- `feverslop.domain` contains render plan, LTX, and postprocessing domain types.
- `feverslop.ports` defines protocols and shared request types; ports do not import adapters.
- `feverslop.adapters` contains ComfyUI, local JSON artifacts, OpenAI-compatible LLM clients, and FFmpeg/postprocessing integration.
- repository-root Python files are public CLI scripts or compatibility facades only.

Allowed root Python files:

- Public CLI scripts: `main.py`, `render_ltx.py`, `render_storyboard.py`, `compact_relay_prompts.py`, `fix_ltx_prompt_anchors.py`, `storyboard_page.py`, `normalize_render_plan.py`, `repair_scene_srt.py`, `trim_existing_ltx_clips.py`.
- Compatibility facades only: `ltx_video_renderer.py`, `storyboard_renderer.py`, `workflow_patcher.py`.

Import policy for new implementation code:

- new implementation imports must use `feverslop.*`.
- no new code should import `application.*`, `adapters.*`, `domain.*`, or `ports.*` from repository-root packages.
- root modules are reserved for CLI entry points and compatibility facades.
- compatibility facades should re-export explicit names and must not use `import *`.
