# CLI and MiniMax H3 Compatibility

This page documents the current compatibility boundaries for the command-line
tools and scene-level MiniMax H3 R2V references.

## CLI implementation and compatibility facades

The repository-root scripts remain supported entry points for existing command
lines and imports. Their canonical implementations live under
`src/feverslop/cli/`:

| Public entry point | Canonical implementation |
| --- | --- |
| `render_ltx.py` | `feverslop.cli.render_ltx` |
| `render_storyboard.py` | `feverslop.cli.render_storyboard` |
| `compact_relay_prompts.py` | `feverslop.cli.compact_relay_prompts` |
| `fix_ltx_prompt_anchors.py` | `feverslop.cli.fix_ltx_prompt_anchors` |
| shared render arguments | `feverslop.cli.shared_args` |

The root files are explicit compatibility facades. They continue to expose the
legacy `main()` functions and public import seams, but new implementation code
should import the package modules instead. The unified `main.py render` parser
and the legacy root-script parsers use the same shared render argument
definitions, so supported options keep the same defaults across both forms.

## Intentional location-only H3 scenes

MiniMax H3 R2V supports scenes that intentionally show only an environment,
without an actor. Mark this at scene level in the render-plan references:

```json
{
  "references": {
    "subject_mode": "location_only",
    "actor_ids": [],
    "location_id": "forest",
    "location_msr_path": "output/references/locations/forest.png"
  }
}
```

For this contract:

- cast resolution does not insert the first available actor;
- stale `actor_sheet_paths` or `actor_msr_paths` are ignored;
- the H3 prompt receives the location image as an `environment` reference;
- MiniMax H3 R2V accepts the scene when a location reference is available;
- a scene with neither an actor nor a location reference is rejected with a
  scene-specific validation error.

`subject_mode` is a scene reference intent, not a replacement for the global
project `subject_mode` setting (`single` or `multi`). Existing actor-led scenes
continue to use the project setting and retain their previous behavior.

Malformed or unknown actor assignments that are reconstructed instead of being
explicitly marked `location_only` emit an English warning containing the scene
number and reconstructed actor IDs. Treat that warning as a data-quality issue
and correct the scene references rather than relying on the fallback.

## Verification

After changing these compatibility boundaries, run the focused tests first and
then the full suite when practical:

```powershell
uv run python -m unittest tests.test_public_compatibility tests.test_scene_cast tests.test_comfyui_minimax_h3_r2v_backend
uv run python -m unittest discover -s tests
ruff check .
```

