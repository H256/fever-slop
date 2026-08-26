# Sequence-to-Sheet Reference Pipeline

The sequence reference pipeline creates reusable actor and location references
from a generated anchor image and a short multi-view video. It is intended for
the R2V and I2V stages of both music-video and Movie projects.

The default reference path remains `image_views`. Opt into the sequence path
with `--reference-generation sequence_sheet`.

When `sequence_sheet` is enabled, the default workflow pair is:

- Anchor image: `workflows/image/image-model/image_t2i_startframe_krea_v1.json` (Krea2 T2I).
- Multi-view sequence: `workflows/sequence/minimax_h3/sequence_to_sheet_minimax_h3_i2va_v1.json`
  (MiniMax H3 I2VA).

Both workflows can be overridden. The anchor workflow is optional in the
examples below because the Krea2 workflow is already the default.

## What it does

```text
anchor image -> MiniMax H3 I2VA sequence -> selected frames
             -> contact sheet -> tiled actor/location sheet
             -> global asset library + MSR/R2V/I2V manifests
```

The video is an intermediate view-generation pass, not the final reference
asset. The pipeline extracts representative frames, removes duplicate frames,
and composes a regular sheet from the selected tiles. Actor sheets use a
portrait-oriented 2x3 layout; location sheets use a landscape-oriented 3x2
layout. The original contact sheet is kept for inspection and later tooling.

## Standalone hero-image import

An existing hero image can be used as the real sequence anchor without
rendering a replacement anchor first:

```powershell
uv run python -m feverslop.tools.generate_sequence_sheet `
  --source-image .\hero.png `
  --kind character `
  --id the_dark_man `
  --description "dark-haired man, black coat, pale skin" `
  --publish local `
  --json
```

The command validates the asset ID and source image before contacting ComfyUI.
It writes `hero.png`, `anchor.png`, `sequence.mp4`, selected frames,
`contact-sheet.png`, `sheet.png`, and an inspectable `manifest.json` below
`references/<kind-directory>/<id>/`. Character imports select six views by
default; locations select five. Use `--publish project --project <dir>` to
write below `<dir>/references`, or `--publish global --library-root <dir>` to
publish the completed look to the global asset library.

Use `--dry-run --json` to validate the source and inspect the planned output
without loading ComfyUI or calling a vision model. If the description is not
known, `--description-mode auto` uses the configured vision-capable LLM and
records its model in the manifest; the hero image remains the visual source.

Generated artifacts are stored below the project output directory:

```text
output/references/actors/<id>/anchor.png
output/references/actors/<id>/sequence.mp4
output/references/actors/<id>/contact-sheet.png
output/references/actors/<id>/sheet.png
output/references/actors/<id>/frames/frame_*.png
```

Movie projects use the equivalent `movie/references/actors/<id>/` and
`movie/references/locations/<id>/` paths. The manifest points R2V/I2V at
`sheet.png`; locations also expose the anchor as `msr_background_path`.

## Music-video tutorial

Generate only the reference and prompt-preparation stages with the sequence
pipeline:

```powershell
uv run python run_pipeline.py .\projects\my-song `
  --video-pipeline minimax-h3-r2v `
  --reference-generation sequence_sheet `
  --sequence-to-sheet-workflow .\workflows\sequence\minimax_h3\sequence_to_sheet_minimax_h3_i2va_v1.json `
  --stage main_pipeline `
  --stage anchor_fix `
  --stage msr_references `
  --stage msr_reference_sheets `
  --stage h3_prompts `
  --stage render_plan `
  --skip-tests
```

After reviewing the generated sheets and the enriched render plan, continue
with the video render command while retaining
`--reference-generation sequence_sheet` and the same sequence workflow. Do not
use an unqualified normal rerun: the compatibility default is `image_views` and
would regenerate a different reference set.

The same option works with the LTX MSR path:

```powershell
uv run python run_pipeline.py .\projects\my-song `
  --video-pipeline ltx_msr `
  --reference-generation sequence_sheet `
  --sequence-to-sheet-workflow .\workflows\sequence\minimax_h3\sequence_to_sheet_minimax_h3_i2va_v1.json `
  --skip-tests
```

## Movie tutorial

Prepare Movie references and the MiniMax H3 plan without queueing the final
movie render:

```powershell
uv run python movie_pipeline.py .\projects\my-movie `
  --movie-video-workflow minimax-h3-r2v `
  --reference-generation sequence_sheet `
  --sequence-to-sheet-workflow .\workflows\sequence\minimax_h3\sequence_to_sheet_minimax_h3_i2va_v1.json `
  --r2v-workflow .\workflows\video\minimax_h3\r2v_v1.json `
  --skip-movie-render
```

Force regeneration when references already exist:

```powershell
uv run python movie_pipeline.py .\projects\my-movie `
  --movie-video-workflow minimax-h3-r2v `
  --reference-generation sequence_sheet `
  --force-movie-references `
  --skip-movie-render
```

To use a different anchor/T2I workflow, add `--reference-hero-workflow` to the
music-video command or `--hero-workflow` to the Movie command. For example:

```powershell
uv run python movie_pipeline.py .\projects\my-movie `
  --movie-video-workflow minimax-h3-r2v `
  --reference-generation sequence_sheet `
  --hero-workflow .\workflows\my_reference_workflow.json `
  --sequence-to-sheet-workflow .\workflows\sequence\minimax_h3\sequence_to_sheet_minimax_h3_i2va_v1.json `
  --skip-movie-render
```

Review `movie/references/manifest.json` and the files below
`movie/references/` before starting the final render. The downstream scene
reference composer still creates per-shot contact sheets from the actor,
location, and prop assets; this feature supplies the canonical actor/location
assets it consumes.

## Direct Reference Bible invocation

For debugging or rebuilding references independently of the full runner:

```powershell
uv run python -m feverslop.tools.reference_bible `
  --project-config .\projects\my-song\config.json `
  --app-config .\app_config.json `
  --hero-workflow .\workflows\my_reference_workflow.json `
  --edit-workflow .\workflows\my_reference_edit_workflow.json `
  --reference-generation sequence_sheet `
  --sequence-workflow .\workflows\sequence\minimax_h3\sequence_to_sheet_minimax_h3_i2va_v1.json
```

Use `uv run python -m feverslop.tools.reference_bible --help` to see the
required project and workflow options for the local configuration.

## Operational notes

- `image_views` is the compatibility default and does not require a sequence
  workflow.
- `sequence_sheet` requires a ComfyUI API workflow compatible with the
  `ComfyUISequenceToSheetBackend`; the bundled MiniMax H3 I2VA workflow is the
  reference template.
- Keep the sequence workflow VRAM-safe and use its cleanup/offload nodes where
  the local ComfyUI installation requires them.
- The sequence render can be expensive. Inspect the generated intermediate
  `sequence.mp4` and `contact-sheet.png` before spending time on downstream
  R2V/I2V rendering.
- DSPy produces the semantic reference-sheet plan in `sequence_sheet` mode.
  The deterministic compiler enforces required views, framing, anchor rules,
  identity constraints, and negative constraints before the backend serializes
  the prompt. If DSPy is unavailable or incomplete, the same compiler receives
  a deterministic fallback plan.
- DSPy remains responsible for prompt enrichment in the supported MiniMax H3
  prompt stages; it is not used to compose the image tiles.

## Benchmarking a replacement

Before replacing `image_views` or `sequence_sheet`, record pinned inputs and
machine-readable measurements in a benchmark JSON file. The evaluator does not
render media and does not commit generated artifacts; it evaluates recorded
runs and retains only artifact/provenance references in the report:

```powershell
uv run python -m feverslop.tools.reference_sheet_benchmark `
  --config .\benchmarks\reference-sheets\2026-08-25.json `
  --report .\benchmarks\reference-sheets\reports\2026-08-25.json
```

Each candidate run must identify its fixture and provide scores for identity
consistency, view coverage, sharpness, layout continuity, and reproducibility.
Failures force the `fallback` decision even when the quality scores pass. A
candidate is recommended for `replace` only when every configured gate passes;
runtime, retry counts, model/workflow revisions, and artifact hashes should be
kept in the candidate's provenance records.

This scaffold intentionally does not claim visual quality, perform human
review, or decide deprecation policy by itself. Those measurements and the
human checklist remain inputs to the pinned report. It also does not include
generated images or videos in Git.

