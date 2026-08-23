# Render Plan Artifacts

Music-video render plans live in `output/render/plans/`. The filenames identify pipeline stages, not schema versions. Do not create chains such as `ingredients_v2_final_fixed.json`.

| File | Owner | Purpose | Recreated by | Retention |
| --- | --- | --- | --- | --- |
| `base.json` | main render-plan generation | Complete authoring plan and base resume point | main pipeline | Keep while the project may be regenerated |
| `compact.json` | relay compaction | Intermediate input for anchor correction | relay compact stage | Disposable after `anchored.json` exists |
| `anchored.json` | anchor correction | Corrected prompts for storyboard/reference enrichment | anchor fix stage | Keep while downstream plans may be regenerated |
| `references.json` | reference and MSR prompt enrichment | Final MSR render plan and source for Ingredients enrichment | MSR reference/prompt stages | Keep for MSR resume and Ingredients regeneration |
| `ingredients.json` | Ingredients sheet enrichment | Compact runtime plan for Ingredients prepare/render | Ingredients sheets stage | Keep for Ingredients resume; regenerate after upstream prompt/reference edits |

## Why `ingredients.json` is compact

The Ingredients renderer needs timing and resolution, actor/location IDs, one composed sheet, one stable global prompt, and one temporal prompt relay. It does not need storyboard prompts, full reference manifests, MSR aliases, or every earlier LTX prompt variant.

The runtime scene therefore keeps:

```text
scene/timing/fps/resolution/cut
metadata.segment_id/type/silent_mode/lyrics
references.actor_ids/location_id
ingredients.sheet_path/anchors/global_prompt
ltx.base_prompt/static_prompt/prompt_relay/native_audio
```

`ltx.prompt_relay` is authoritative for V4 workflows. `ltx.static_prompt` is a deterministic compatibility summary for explicitly selected V3 workflows. V3 remains renderable, but its singing and instrumental transitions are best effort because a single `#PROMPT_POSITIVE` conditioning value cannot enforce frame boundaries.

## Prepared scene artifacts

Each `output/render/scenes/scene_NNNN/` directory contains:

- `manifest.json`: hashes the selected workflow template, projected scene, assets, seed, and render settings. Prepare uses it to decide whether cached output is still valid.
- `workflow.json`: fully patched ComfyUI API workflow. It is generated output and is overwritten by prepare.
- `raw.mp4`: downloaded ComfyUI result before rolling-window trimming.
- `final.mp4`: scene clip after trimming.

Do not treat `workflow.json` as a planning source. Edit the appropriate plan and rerun prepare.

## Editing and regeneration

- Edit `base.json` or `anchored.json` when changing cast, location, concept, or source prompts and then rerun downstream enrichment.
- Edit `references.json[].ltx.msr_prompt_relay` only when intentionally resuming after MSR enrichment.
- Edit `ingredients.json[].ltx.prompt_relay` for a targeted V4 rerender when upstream regeneration is not desired.
- Deleting `compact.json` after `anchored.json` exists is harmless.
- Deleting `references.json` requires reference/MSR enrichment before MSR or Ingredients preparation.
- Deleting `ingredients.json` requires the Ingredients sheet stage before Ingredients preparation.
- Changes to the projected scene or selected workflow invalidate its prepared manifest and require prepare before render.

Generated project outputs are local artifacts and must not be committed.
