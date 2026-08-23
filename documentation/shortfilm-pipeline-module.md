# Shortfilm Pipeline Module Concept

Status: Proposal; the shortfilm module described here is not implemented in
the current source tree.

This document describes a proposed FeverSlop extension for generating short films instead of music videos.

The goal is not to bend the existing music-video pipeline into a different shape. The goal is to add a dedicated shortfilm pipeline module that reuses the existing infrastructure where it fits:

- OpenAI-compatible LLM client
- JSON artifact store
- ComfyUI client
- ComfyUI model resolver
- workflow patching
- MSR actor/location reference generation
- MSR reference manifests
- FFmpeg utilities

The shortfilm module should have its own story, screenplay, shot-plan, and AV-rendering concepts.

## Current Music-Video Model

The current pipeline is audio-first:

```text
song/audio + lyrics
  -> audio timeline
  -> vocal/instrumental segments
  -> scene prompts
  -> render plan
  -> LTX clips
  -> video concat
  -> original song audio mux
```

For music videos, the audio file is the source of truth. Scene durations, singing ranges, and final muxing all follow the song.

That does not match short films.

## Shortfilm Model

The shortfilm pipeline should be story-first:

```text
idea + style + duration + resolution
  -> story bible
  -> characters and locations
  -> screenplay
  -> shot plan
  -> MSR references
  -> MSR film plan
  -> LTX MSR AV clips
  -> final concat preserving generated audio
```

For short films, the story and shot plan are the source of truth. LTX/MSR is allowed to generate the scene audio, including dialogue, ambience, and foley, so the final assembly must preserve clip audio instead of muxing a song over the result.

## Module Boundary

The recommended shape is a dedicated shortfilm module:

```text
src/feverslop/shortfilm/
|-- domain/
|   |-- story_bible.py
|   |-- screenplay.py
|   |-- shot_plan.py
|   |-- msr_film_plan.py
|   |-- characters.py
|   |-- dialogue.py
|   `-- sound.py
|-- application/
|   |-- autoproduce_config.py
|   |-- generate_story_bible.py
|   |-- generate_screenplay.py
|   |-- generate_shot_plan.py
|   |-- validate_shot_plan.py
|   |-- build_msr_film_plan.py
|   |-- enrich_msr_references.py
|   |-- render_shortfilm.py
|   `-- assemble_shortfilm.py
|-- ports/
|   |-- story_generation.py
|   |-- av_rendering.py
|   |-- artifact_store.py
|   `-- postprocessing.py
|-- adapters/
|   |-- llm_story_generator.py
|   |-- comfyui_msr_av_backend.py
|   `-- ffmpeg_shortfilm_assembler.py
`-- composition/
    `-- shortfilm_runner.py
```

If keeping all implementation under the existing top-level layers is preferred, the same boundaries can be represented as:

```text
src/feverslop/domain/shortfilm_*.py
src/feverslop/application/shortfilm_*.py
src/feverslop/ports/shortfilm_*.py
src/feverslop/adapters/comfyui_shortfilm_*.py
src/feverslop/composition/shortfilm_runner.py
```

The important part is the boundary, not the exact package layout.

## Autoproduce Mode

Autoproduce should not be a separate rendering path.

Autoproduce should only fill the project config and intermediate planning artifacts. After that, the normal modular shortfilm pipeline steps should run.

Minimal autoproduce input:

```json
{
  "project_name": "door_below",
  "mode": "autoproduce",
  "duration_seconds": 90,
  "video": {
    "fps": 24,
    "width": 1920,
    "height": 1088
  },
  "idea": "A repair technician finds a locked door under an abandoned hotel.",
  "style": "contained supernatural thriller, realistic cinematic lighting, slow tension, no comedy",
  "language": "en",
  "constraints": {
    "max_characters": 2,
    "max_locations": 2,
    "max_actors_per_shot": 2,
    "max_shot_duration_seconds": 8,
    "dialogue_density": "low"
  }
}
```

Autoproduce should then create or update:

```text
projects/my_shortfilm/
|-- config.json
|-- shortfilm_config.json
`-- output/
    `-- shortfilm/
        |-- story_bible_<id>.json
        |-- screenplay_<id>.json
        |-- shot_plan_<id>.json
        `-- msr_film_plan_<id>.json
```

The generated `config.json` should contain the production data that existing or shared pipeline components can consume:

```json
{
  "project_name": "door_below",
  "duration_seconds": 90,
  "story_idea": "A repair technician finds a locked door under an abandoned hotel.",
  "style": "contained supernatural thriller, realistic cinematic lighting, slow tension, no comedy",
  "video": {
    "fps": 24,
    "width": 1920,
    "height": 1088
  },
  "actors": [],
  "locations": []
}
```

Autoproduce fills `actors` and `locations` after generating the story bible.

## Story Bible

The story bible is the stable production source for consistency. It should be generated before screenplay or shot planning.

Example:

```json
{
  "title": "The Door Below",
  "genre": "supernatural thriller",
  "logline": "A repair technician discovers a locked door under an abandoned hotel and hears her own voice behind it.",
  "tone": "quiet dread, grounded realism, minimal dialogue",
  "runtime_seconds": 90,
  "visual_rules": [
    "single practical light source",
    "damp concrete textures",
    "no exterior city shots"
  ],
  "continuity_rules": [
    "Mara always wears the same yellow work jacket",
    "the red utility lamp remains the main background color accent"
  ],
  "characters": [
    {
      "id": "mara",
      "name": "Mara",
      "role": "repair technician",
      "visual_description": "A tired repair technician in her late twenties with short dark hair, a yellow work jacket, gray cargo pants, and a compact tool belt.",
      "wardrobe": "yellow work jacket, gray cargo pants, black work boots, compact tool belt",
      "voice": "low, controlled, tired",
      "continuity_rules": [
        "Always wears the yellow work jacket",
        "Never changes hairstyle",
        "Carries the same compact tool belt"
      ]
    }
  ],
  "locations": [
    {
      "id": "hotel_basement",
      "name": "Hotel Basement",
      "visual_description": "A narrow abandoned hotel basement with damp concrete walls, exposed pipes, an old electrical panel, and a locked metal door at the end of the corridor.",
      "lighting": "dim practical work light, red utility lamp near the door, deep shadow",
      "continuity_rules": [
        "Concrete walls remain damp",
        "The locked metal door stays at the end of the corridor",
        "The red utility lamp remains visible near the door"
      ]
    }
  ]
}
```

The story bible should be the input for reference generation. Once MSR references are created, character and location IDs are stable and later planning steps must use only existing IDs.

## Screenplay

The screenplay represents dramatic structure, action, and dialogue before it is split into renderable shots.

Example:

```json
{
  "scenes": [
    {
      "scene": 1,
      "location_id": "hotel_basement",
      "dramatic_purpose": "Introduce the locked door and Mara's isolation.",
      "action": "Mara checks an electrical panel, hears a knock from behind the locked door, and realizes the sound matches her own tapping rhythm.",
      "dialogue": [
        {
          "speaker_id": "mara",
          "text": "That was not in the plans.",
          "delivery": "whispered, tense"
        }
      ]
    }
  ]
}
```

The screenplay does not need to be directly renderable. It is an editable planning artifact.

## Shot Plan

The shot plan is the renderable story structure. It should contain exact durations, actor IDs, location IDs, visual action, camera intent, dialogue, and sound intent.

Example:

```json
{
  "target_duration_seconds": 90,
  "shots": [
    {
      "shot": 1,
      "duration_seconds": 6.0,
      "actor_ids": ["mara"],
      "location_id": "hotel_basement",
      "visual_action": "Mara kneels beside the electrical panel and turns toward the locked metal door.",
      "camera": "slow handheld push-in, medium close-up",
      "dialogue": [
        {
          "speaker_id": "mara",
          "text": "That was not in the plans.",
          "delivery": "whispered, tense"
        }
      ],
      "sound": {
        "ambience": "low electrical hum, distant water drops",
        "foley": "soft fabric movement, metal panel creak",
        "music": "none"
      }
    }
  ]
}
```

The shot plan should be validated before rendering:

- total duration is close to the requested runtime
- every actor ID exists in the story bible/config
- every location ID exists in the story bible/config
- no shot exceeds `max_shot_duration_seconds`
- no shot exceeds `max_actors_per_shot`
- dialogue speaker IDs exist and are present in the shot actor IDs unless intentionally offscreen
- sound fields are present
- visual action does not introduce undefined characters or locations

If validation fails, a repair step should ask the LLM to fix only the invalid fields.

## Duration Budgeting

Autoproduce must treat the requested duration as a hard production constraint.

For example:

```text
target duration: 90s
shot count: 12-16
min shot duration: 3s
max shot duration: 8s
accepted total duration tolerance: +/- 2s
```

The shot-plan generator should produce durations whose sum is inside the tolerance. The validator should reject plans that drift too far.

## MSR Reference Generation

The existing reference generator should be reused and generalized where necessary.

The shortfilm pipeline should feed it the story bible or the generated `config.json` actors and locations. It should continue to write the same reference artifacts:

```text
output/references/actors/<actor_id>/manifest.json
output/references/actors/<actor_id>/sheet.png
output/references/actors/<actor_id>/msr_sheet.png
output/references/locations/<location_id>/manifest.json
output/references/locations/<location_id>/views/hero.png
```

For MSR shortfilms, the important manifest fields are:

```text
actors:
  id
  name
  sheet_path
  msr_input_path

locations:
  id
  name
  sheet_path
  msr_background_path
```

The shortfilm module should not create a separate reference asset system unless the existing one becomes impossible to generalize.

## MSR Film Plan

The MSR film plan is the renderer-facing artifact. It is built from:

- shot plan
- story bible
- reference manifests
- video settings

Example:

```json
{
  "scene": 1,
  "duration_seconds": 6.0,
  "fps": 24,
  "width": 1920,
  "height": 1088,
  "frame_count": 144,
  "references": {
    "actor_ids": ["mara"],
    "location_id": "hotel_basement",
    "actor_msr_paths": [
      "output/references/actors/mara/msr_sheet.png"
    ],
    "location_msr_path": "output/references/locations/hotel_basement/views/hero.png"
  },
  "msr": {
    "visual_prompt": "Mara kneels beside an old electrical panel in the abandoned hotel basement, then turns toward the locked metal door at the end of the corridor.",
    "dialogue_prompt": "Mara whispers, tense and controlled: \"That was not in the plans.\"",
    "sound_prompt": "Low electrical hum, distant water drops, soft fabric movement, a subtle metal panel creak. No music.",
    "continuity_prompt": "Use reference actor 1 as Mara. Preserve her short dark hair, yellow work jacket, gray cargo pants, black work boots, and compact tool belt. Use the basement background reference for the location.",
    "negative_prompt": "Do not introduce new characters, new locations, different wardrobe, or a different hairstyle."
  },
  "metadata": {
    "shot_type": "dialogue",
    "speaker_ids": ["mara"]
  }
}
```

The MSR film plan should keep visual, dialogue, sound, continuity, and negative prompts separate. The concrete ComfyUI adapter can map them into the chosen workflow.

Avoid forcing shortfilm prompts into music-video fields such as `ltx.base_prompt` or `ltx.prompt_relay` unless a compatibility adapter explicitly needs that shape.

## AV Rendering Port

The current video render request requires an external `audio_file`, which is correct for music videos but wrong for shortfilms where the model may generate audio.

Add a shortfilm AV rendering port:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class AVRenderRequest:
    scene: dict
    scene_number: int
    workflow_path: Path
    output_dir: Path
    generate_audio: bool = True
    external_audio_file: Path | None = None


@dataclass(frozen=True)
class AVRenderResult:
    scene_number: int
    clip_path: Path
    has_audio: bool
    duration_seconds: float


class AVRenderBackend(Protocol):
    def render_av(self, request: AVRenderRequest) -> AVRenderResult:
        """Render one shortfilm shot, preserving generated audio when present."""
```

## ComfyUI MSR AV Adapter

Add a dedicated adapter instead of extending the song-oriented MSR backend too far:

```text
src/feverslop/shortfilm/adapters/comfyui_msr_av_backend.py
```

The adapter should patch anchors such as:

```text
#MSR_ACTOR_1
#MSR_ACTOR_2
#MSR_ACTOR_3
#MSR_ACTOR_4
#MSR_BACKGROUND
#PROMPT_VISUAL
#PROMPT_DIALOGUE
#PROMPT_SOUND
#PROMPT_CONTINUITY
#PROMPT_NEGATIVE
#FRAMES
#WIDTH
#HEIGHT
#FRAMERATE
#SAVE_VIDEO
```

The exact workflow contract can evolve, but it should support:

- actor MSR reference images
- location/background MSR reference image
- visual prompt
- dialogue prompt
- sound prompt
- continuity prompt
- frame count
- resolution
- FPS
- generated audio output

The adapter should not require a song audio file.

## Assembly

Music-video assembly currently creates a video-only concat and then muxes the original full song audio.

Shortfilm assembly should preserve generated clip audio:

```text
rendered AV clips
  -> concat preserving audio
  -> optional loudness normalization
  -> optional subtitles
  -> final shortfilm MP4
```

Minimal assembly can use FFmpeg concat:

```bash
ffmpeg -f concat -safe 0 -i concat_list.txt -c copy final.mp4
```

Later improvements:

- reencode for consistent codecs
- loudness normalization
- room-tone handling
- audio crossfades
- subtitle muxing from dialogue fields
- diagnostic scene manifest with detected duration and audio presence

## CLI Shape

Two-step mode:

```bash
uv run python shortfilm_autoproduce.py projects/door_below \
  --duration-seconds 90 \
  --width 1920 \
  --height 1088 \
  --idea "A repair technician finds a locked door under an abandoned hotel." \
  --style "contained supernatural thriller, realistic cinematic lighting"

uv run python shortfilm_pipeline.py projects/door_below
```

Combined mode:

```bash
uv run python shortfilm_pipeline.py projects/door_below \
  --autoproduce \
  --duration-seconds 90 \
  --width 1920 \
  --height 1088 \
  --idea "A repair technician finds a locked door under an abandoned hotel." \
  --style "contained supernatural thriller, realistic cinematic lighting"
```

Internally, `--autoproduce` should only run the config/artifact bootstrapper first. Rendering should still go through the same shortfilm pipeline steps.

## Manual Edit Workflow

The design should allow editing after any major artifact:

```text
shortfilm_config.json
story_bible_<id>.json
screenplay_<id>.json
shot_plan_<id>.json
msr_film_plan_<id>.json
```

Examples:

- Change a character in the story bible, regenerate references, rebuild the MSR film plan.
- Edit the screenplay, regenerate the shot plan.
- Edit the shot plan, rebuild only the MSR film plan.
- Edit the MSR film plan, rerender selected shots.
- Keep references stable while changing story action.

Autoproduce should be a convenience layer, not an opaque one-shot mode.

## Reuse From Existing Pipeline

Reuse directly:

- `ComfyUIClient`
- `ComfyUIModelResolver`
- `WorkflowPatcher`
- `JsonArtifactStore`
- OpenAI-compatible LLM adapter
- Reference-Bible rendering and manifest structure
- MSR actor/location reference conventions
- FFmpeg helper functions where applicable

Reuse with adaptation:

- reference-bible input extraction, generalized from music-video context to story-bible/config context
- MSR reference enrichment, generalized from render-plan scenes to shortfilm shots
- postprocessing utilities, extended for AV concat and loudness normalization

Avoid reusing directly:

- audio timeline generation
- beat analysis
- vocal/instrumental segment generation
- lyrics alignment
- `ltx_prompt_relay`
- song-based original-audio muxing
- rolling song audio window logic

## MVP Implementation Order

1. Define `shortfilm_config.json` and story-bible schema.
2. Implement autoproduce config bootstrapper.
3. Generate `story_bible_<id>.json`, `screenplay_<id>.json`, and `shot_plan_<id>.json`.
4. Add shot-plan validation and repair.
5. Generalize reference generation to consume shortfilm actors and locations.
6. Build `shot_plan -> msr_film_plan`.
7. Add `AVRenderBackend` port.
8. Implement `ComfyUIMSRAVBackend` for one concrete MSR workflow.
9. Add AV concat that preserves generated clip audio.
10. Add CLI entry points.

## Design Principle

Autoproduce fills inputs and planning artifacts.

The pipeline renders from explicit JSON artifacts.

MSR references are generated once and reused through stable actor and location IDs.

The shortfilm pipeline stays separate from the music-video pipeline, while sharing the infrastructure that is genuinely common.
