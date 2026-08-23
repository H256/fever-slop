# Global Asset Library

FeverSlop keeps canonical characters, locations, styles, and props outside a
project. The default library is `~/.feverslop/library/`; set
`global_library_path` in the app config to use another location. Every asset
has a stable ID, a validated `manifest.json`, and an integer revision.

## Layout and configuration

```text
~/.feverslop/library/
  character/ava/manifest.json
  character/ava/hero.png
  prop/guitar/manifest.json
```

Project `config.json` can opt in without changing legacy `actors` or
`locations` fields:

```json
{
  "global_cast": [{"asset_id": "ava", "look_id": "default", "role": "lead"}],
  "global_locations": [{"asset_id": "nightclub", "look_id": "default"}],
  "global_styles": [{"asset_id": "neon-noir"}],
  "global_props": [{"asset_id": "guitar", "look_id": "default"}]
}
```

## CLI

Use the `feverslop global-library` command (or
`python -m feverslop.tools.global_library_cli`):

```text
feverslop global-library list --json
feverslop global-library create --kind character --id ava --name Ava
feverslop global-library create-look --kind character --id ava --look-id default --name Default
feverslop global-library show --kind character --id ava --json
feverslop global-library validate
feverslop global-library delete --kind prop --id guitar
feverslop global-library generate --kind character --id ava --name Ava \
  --idea "silver bob and black leather jacket" --workflow character-sheet-v1 --dry-run
```

`generate --input idea.json` accepts the same fields as the typed intake. Use
`--input -` for JSON on stdin, or `--interactive` for missing fields. A
workflow profile must be named explicitly; `--dry-run` prints the normalized
request before any generation.

## Snapshots, revisions, and portability

Resolution copies selected media and the source revision into
`output/references/global_assets/<kind>/<id>/<look>/`. Rendering uses these
project-local files, so the project remains usable if the global library is
offline. Changes in the library never rewrite an existing project silently.
Run an explicit refresh after reviewing a stale snapshot. Optimistic revision
checks reject updates based on an old manifest, and concurrent writers are
serialized per asset.

Invalid kinds, duplicate looks, unsafe paths, missing IDs/looks, invalid
manifests, missing library roots, and workflow failures return actionable
errors. Generator runs preserve the normalized request, selected workflow,
outputs, and status under the library `runs/` directory so completed runs can
be resumed.

## Prop interactions

Scenes may list `prop_ids` and typed interactions such as:

```json
{"actor_id": "ava", "prop_id": "guitar", "action": "holds", "relationship": "instrument"}
```

Validation rejects unknown actors or props and keeps scenes without props
unchanged. Prompt projections retain stable IDs and interaction semantics.
