# Cast brief and roster policy

Projects may define a free cast brief independently from `story_idea` and
`style`:

```json
{
  "cast_idea": "Four grotesque musicians; the drummer is massive.",
  "cast_policy": {"mode": "extend", "target_size": 4},
  "actors": [{"id": "drummer", "name": "Drummer", "role": "drummer"}]
}
```

`extend` preserves configured actors and may add generated members. `fixed`
only enriches configured members. `target_size` is the total roster size; it
is independent of `max_scene_actors`, which only limits visible actors in an
individual scene. Explicit actor fields take precedence over the free brief.
