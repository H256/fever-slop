# Prompt Architecture

Status: Current reference

FeverSlop prompt-writing stages use DSPy signatures and typed Pydantic payloads.
The DSPy module owns the model call, structured output, guide selection, and
timeout configuration. Production code should not build a second inline prompt
for a migrated stage.

## Bundled guide naming

Guides are Markdown resources in `feverslop.prompting.guides`. Names are
lowercase, hyphen-separated stems; callers may pass either the stem or the
`.md` filename to `load_markdown_guide()`.

The prefix identifies the pipeline and the suffix identifies the contract:

- `music-video-*`, `movie-*`, `msr-*`, `relay-directions`, and
  `ingredients-vision` are stage guides.
- `minimax-h3-base.md` covers T2V, I2V, FL2V, and L2V.
- `minimax-h3-references.md` covers R2V reference roles.
- `krea-actor.md` and `krea-location.md` are injected only for Krea image
  reference workflows.

All Markdown files under `src/feverslop/prompting/guides` are package data and
are checked in both wheel and source-distribution builds.

## Signatures and structured inputs

Each DSPy module exposes a named signature with a `guide` input, a typed
`payload` input, and a typed `result` output. Movie planning uses ten
signatures: story arch, movie bible, actor refinement, location refinement,
continuity plan, story design, screenplay, narrative plan, shot plan from
bible, and shot plan.

Payloads contain domain values such as `StoryArch`, `MovieBible`, actors,
locations, screenplay scenes, shots, and configuration dictionaries. JSON
strings are not used as the primary contract. H3 generation similarly passes
typed references, relay segments, audio references, and scene context; the
generated prompt remains the output of the DSPy contract.

## Fallbacks and concurrency

Fallbacks are narrow and stage-specific. Movie actor/location refinement keeps
the original domain objects when the DSPy call fails or returns unusable data;
shot planning falls back to the deterministic planner when no shots are
returned. These fallbacks preserve an artifact-producing pipeline, but do not
silently restore the removed Jinja prompt builders.

The configured `llm.max_concurrent_requests` value is a process-local ceiling
shared by direct OpenAI-compatible requests and DSPy/LiteLLM requests. The
default is one. It does not coordinate separate FeverSlop processes or other
clients using the same server.

## Future optimization hooks

The module boundary leaves room for DSPy optimization or compilation without
changing pipeline callers: predictors are created from signature bundles,
guides are loaded by name, payloads are validated before prediction, and
timeouts are passed as DSPy LM configuration. A future optimizer can therefore
attach compiled predictors or a cache at the module/runtime boundary while
keeping guide names, payload models, result models, and fallback behavior
stable.

## Guide ownership matrix

This is the practical edit map. Change the Markdown guide when changing model
instructions or examples. Change the corresponding signature/module when
changing input or output fields, validation, or the call contract. The
application call site should only assemble domain data and consume the typed
result.

| Guide(s) | Module / signature | Main call site | Contract |
| --- | --- | --- | --- |
| `music-video-story-idea.md`, `music-video-style.md`, `music-video-subject-locations.md` | `music_video_modules.py` / `music_video_signatures.py` | `prompt_pipeline.py` | Story idea, style block, actors and locations |
| `music-video-concepts.md`, `music-video-concept-repair.md`, `music-video-summary.md` | `music_video_modules.py` / `music_video_signatures.py` | `concept_prompt_batcher.py`, `prompt_pipeline.py` | Timed concept map, missing-key repair, continuity summary |
| `music-video-detail.md`, `music-video-t2i.md`, `music-video-i2v.md` | `music_video_modules.py` / `music_video_signatures.py` | `music_video_prompt_style.py`, `prompt_pipeline.py` | Detail, text-to-image and image-to-video prompts |
| `movie-story-arch.md`, `movie-bible.md`, `movie-refine-actors.md`, `movie-refine-locations.md` | `movie_planning_modules.py` / `movie_planning_signatures.py` | `adapters/movie_planning.py` | Story arch, bible, actor and location refinement |
| `movie-continuity-plan.md`, `movie-story-design.md`, `movie-screenplay.md`, `movie-narrative-plan.md` | `movie_planning_modules.py` / `movie_planning_signatures.py` | `adapters/movie_planning.py` | Continuity, story design, screenplay and narrative plan |
| `movie-shot-plan-bible.md`, `movie-shot-plan.md` | `movie_planning_modules.py` / `movie_planning_signatures.py` | `adapters/movie_planning.py` | Shot plans from bible or scene/story inputs |
| `minimax-h3-base.md`, `minimax-h3-references.md` | `dspy_h3_generator_core.py` / `dspy_h3_signatures.py` | `dspy_h3_prompt_builder.py`, H3 pipeline | T2V/I2V/FL2V/L2V and R2V prompt generation |
| `krea-actor.md`, `krea-location.md` | Krea guide selection in `adapters/movie_planning_prompts.py` | Movie actor/location refinement | Workflow-specific Krea reference rules |
| `msr-vision.md`, `msr-segments.md` | `msr_modules.py` / `msr_signatures.py` | `application/msr_prompt_enrichment.py`, `application/movie_msr_enrichment.py` | Image-grounded references and local MSR directions |
| `relay-directions.md` | `relay_modules.py` / `relay_signatures.py` | `relay_direction_builder.py` | Typed PromptRelay directions and singing constraints |
| `ingredients-vision.md` | `ingredients_modules.py` / `ingredients_signatures.py` | `application/ingredients_vision_prompt.py` | Multimodal reference descriptions and image prompts |
| `song-brief.md`, `lyric-alignment.md` | `general_modules.py` / `general_signatures.py` | `adapters/llm_song_brief_generator.py`, lyric alignment adapter | ACE-Step song brief and transcription correction |
| `storyboard-transform.md` | `general_modules.py` / `general_signatures.py` | `storyboard_prompt_transformer.py` | Typed storyboard prompt transformation |

### What is intentionally not in a guide

The following remain code contracts and should not be moved into prose guides:

- Pydantic schema validation and required fields.
- Image/reference existence checks and reference-role assignment.
- Word limits, safety repairs, singing/lip-sync invariants and silent-mode rules.
- Relay timing, artifact shapes, deterministic fallbacks and file naming.

### Parity status

The H3, Movie, classic Music-Video, MSR, Relay, Ingredients, song-brief and
lyric-alignment guides contain the migrated instruction rules, with template
variables represented by typed payload fields. They are not byte-for-byte
copies because Jinja interpolation, JSON transport instructions and
deterministic validation now live in the signature/module or application
layer. The former private song-brief `_system_prompt()` and the old lyric
alignment constant were removed so each active instruction set has one
editable Markdown source.
