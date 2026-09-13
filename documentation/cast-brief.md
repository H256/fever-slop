# Cast briefs and roster policies

Use a cast brief when you know the kind of recurring characters you want but
do not want to define every visual detail yourself. It is separate from
`story_idea` and `style`, so character requirements do not become accidental
location or style instructions.

The pipeline resolves the cast once into the project context. Scenes select
from that roster; they do not need to show every member, and a narrative scene
may show no cast member at all.

## Choose an input style

Use `actors` for facts that must not change: stable IDs, names, roles, gender,
or a finished appearance. Use `cast_idea` for loose roles, relationships,
appearance direction, or a desired total count. Both can be used together.

```json
{
  "cast_idea": "Four grotesque musicians. The drummer is massive.",
  "cast_policy": {"mode": "extend", "target_size": 4},
  "actors": [
    {"id": "drummer", "name": "Drummer", "role": "drummer"}
  ]
}
```

In this example, `drummer`, its name, and its role are fixed. The model may
fill its missing visual fields and create the remaining three members.

## Roster modes

`cast_policy.mode` controls whether configured actors describe the whole
roster or just anchors.

| Mode | What the resolver may do |
| --- | --- |
| `extend` | Preserve configured actors, fill their missing fields, and add members needed by the brief or target size. |
| `fixed` | Preserve configured actors and fill only missing fields. It must not add members to a configured list. |

`extend` is the default for new policy configurations. Old configurations
without `cast_idea` or `cast_policy` retain their previous behavior.

`target_size` is optional and means the number of actors in the whole resolved
roster. It is not the same as `max_scene_actors`, which caps only the number
of actors visible in one scene.

```json
{
  "cast_policy": {"mode": "fixed", "target_size": 4},
  "actors": [
    {"id": "vocalist", "name": "Mara", "role": "vocalist"},
    {"id": "guitar", "name": "Niko", "role": "guitarist"},
    {"id": "bass", "name": "Iris", "role": "bassist"},
    {"id": "drums", "name": "Jon", "role": "drummer"}
  ],
  "max_scene_actors": 2
}
```

This describes a four-person band but allows scenes with one or two visible
members only.

## Precedence and validation

Explicit non-empty actor fields win over a free brief. For example, a
configured `role: "drummer"` remains a drummer even if the brief is vague.
Actor IDs must be unique. The resolver rejects an empty or duplicate resolved
ID and rejects a roster whose resolved size differs from `target_size` before
reference generation starts.

Do not use `fixed` with a partial list plus `target_size: 4` if you expect the
pipeline to invent three members; that is contradictory. Use `extend` for
that case, or provide all four actors for a fixed roster.

## Full Auto CLI

Full Auto writes the cast brief and policy into the generated project
`config.json`:

```powershell
uv run python -m feverslop.cli.full_auto `
  --idea "A band escapes a collapsing carnival" `
  --style "surreal practical effects" `
  --cast-idea "Four grotesque musicians; a massive drummer" `
  --cast-mode extend `
  --cast-size 4
```

The CLI options map directly to `cast_idea`, `cast_policy.mode`, and
`cast_policy.target_size`. `--cast-size` does not change the per-scene actor
limit.

## Inspect the resolved roster

After prompt generation, inspect
`output/prompts/resolved_context_<song>.json`. Its `actors`, `cast_policy`,
and `cast_contract` fields show the final stable IDs and resolved roster size.
Treat this file as generated output: edit `config.json`, then rerun the
relevant prompt/reference stages when you want to change the source cast.
