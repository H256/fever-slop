# Workflow Import, Ensemble Constraints, and Environment Style

This page documents three project-level features that are otherwise only
discoverable from `--help` or source: the per-project ComfyUI workflow import
(`feverslop workflow-import`), ensemble (group-cast) constraints, and the
`environment_style` configuration key.

## Workflow Import

`workflow-import` manages per-project ComfyUI workflow snapshots. A workflow
file is imported as a draft, validated by a deterministic inspector, gated by a
successful test-run, and then activated. The snapshot state is stored per
project and survives across runs.

The command is `feverslop workflow-import <command>` (also reachable through the
same CLI entry points as the other FeverSlop commands). Every subcommand takes
`--project-dir` (the project directory) and, except `list`, `--profile-id`
(a local profile identifier).

| Subcommand | Purpose |
| --- | --- |
| `import` | Import a workflow file as a draft snapshot. |
| `validate` | Run the deterministic inspector and record the result. |
| `test-run` | Record a test-run outcome for the activation gate. |
| `activate` | Activate a profile, gated by a successful validation **and** a successful test-run. |
| `deactivate` | Deactivate an active profile. |
| `list` | List imported workflow profiles and their state. |

`import` additionally requires `--pipeline` (the pipeline family), `--purpose`
(`preview` or `final`), and `--workflow` (path to the workflow JSON).
`test-run` records the outcome with `--success` / `--fail` and an optional
`--detail` note.

Example:

```powershell
feverslop workflow-import import --project-dir ./projects/my_song --profile-id my_profile --pipeline ltx_i2v --purpose final --workflow ./workflows/video/ltx_25/i2v/i2v_draft.json
feverslop workflow-import validate --project-dir ./projects/my_song --profile-id my_profile
feverslop workflow-import test-run --project-dir ./projects/my_song --profile-id my_profile --success
feverslop workflow-import activate --project-dir ./projects/my_song --profile-id my_profile
feverslop workflow-import list --project-dir ./projects/my_song
```

The inspector is family-agnostic; each pipeline family supplies its boundary as
data in `workflow_import_families.py`. Supported families include
`minimax_h3`, `ltx_ingredients`, `ltx_msr`, and `ltx_i2v`. `krea` is a
non-workflow family and has no workflow boundary.

## Ensemble Constraints

An ensemble is a named, fixed set of actor IDs that must all be present together
in scenes of the configured `required_scene_types`. Narrative scenes (whose type
is not in `required_scene_types`) may use any subset of the ensemble. Missing
members are never silently added; validation returns the missing member IDs so
the caller can emit an explicit message.

Configure ensembles in the project `config.json`:

```json
{
  "ensembles": [
    {
      "id": "chorus_quartet",
      "members": [
        {"actor_id": "hero", "role": "lead"},
        {"actor_id": "backup_vocals", "role": "backup"}
      ],
      "required_scene_types": ["performance"]
    }
  ]
}
```

Each ensemble requires a unique `id` and at least one member with a unique
`actor_id`. `members` entries may include an optional `role`. `required_scene_types`
is a list of scene types (matched case-insensitively) in which every member
must be present.

The constraint is enforced by the visual-consistency preflight
(`_check_ensemble_bindings`). For a scene whose type is in an ensemble's
`required_scene_types`, any missing member produces an `ensemble_incomplete`
issue naming the missing actor IDs. Missing members are not added automatically.

## Environment Style

`environment_style` is an optional project configuration key that sets a
style applied specifically to environment (location) references. It is
consumed by the sequence reference pipeline and the reference bible.

When set, it overrides the global `style` for environment references; when
empty or omitted, the global `style` is used for backward compatibility.

```json
{
  "style": "dark cinematic fantasy, natural textures",
  "environment_style": "moody foggy forest, desaturated green palette"
}
```

Use `environment_style` when the environment should look different from the
overall project style. Leave it empty to keep environment references on the
global style.
