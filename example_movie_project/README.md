# Example Movie Project

One-minute MiniMax H3 R2V short-film test project with six 10-second scenes.

## Story

Maintenance engineer Mara boards a dead orbital relay and races to restore a
distress signal from a trapped survivor capsule. Scenes 3 and 6 contain spoken
dialogue; the other scenes test silent physical performance and environmental
sound.

## Run

From the FeverSlop repository root:

```powershell
uv run python main.py movie example_movie_project `
  --movie-video-workflow minimax-h3-r2v `
  --r2v-workflow workflows/video/minimax_h3/r2v_audio_two_pass.json
```

The first run generates the actor and location reference assets before H3
prompt preparation and rendering. It uses the LLM and ComfyUI endpoints from
`app_config.json`.

To prepare only selected scenes, append for example:

```powershell
--scenes 1,3,6
```

Expected project contract:

- 6 scenes
- 60 seconds total
- 1280 x 704 at 24 fps
- one referenced actor: Mara
- one referenced location: Abandoned Orbital Relay
- MiniMax H3 R2V two-pass workflow
