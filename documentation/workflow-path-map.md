# Workflow asset path map

Status: Inventory baseline for Milestone 22 (#738)

Generated from the tracked workflow JSON files on 2026-08-26. The classification is a review aid, not an authority derived only from filenames; before moving a file, inspect its node graph and all callers. Files under `workflows/old/` are explicitly marked for retirement/review and must not be migrated automatically.

| Current path | Medium | Model family | Mode | Planned target |
|---|---|---|---|---|
| `workflows/audio_song_v2.json` | audio | audio-model | other | workflows/audio/audio-model/ |
| `workflows/image_detail_easyuse_startframe_v1.json` | image | image-model | other | workflows/image/image-model/ |
| `workflows/image_edit_flux2_klein_1ref_v1.json` | image | image-model | edit | workflows/image/image-model/ |
| `workflows/image_edit_flux2_klein_2ref_v1.json` | image | image-model | edit | workflows/image/image-model/ |
| `workflows/image_mask_sam3_actor_regions_v1.json` | image | image-model | mask | workflows/image/image-model/ |
| `workflows/image_repair_sdxl_ipadapter_identity_v1.json` | image | image-model | repair | workflows/image/image-model/ |
| `workflows/image_t2i_startframe_ideogram_director_v1.json` | image | image-model | t2i | workflows/image/image-model/ |
| `workflows/image_t2i_startframe_ideogram_v1.json` | image | image-model | t2i | workflows/image/image-model/ |
| `workflows/image_t2i_startframe_krea_v1.json` | image | image-model | t2i | workflows/image/image-model/ |
| `workflows/image_t2i_startframe_v1.json` | image | image-model | t2i | workflows/image/image-model/ |
| `workflows/old/audio_song.json` | audio | audio-model | other | retire-or-review |
| `workflows/old/autoprompt_image_z_image_turbo.json` | other | other | other | retire-or-review |
| `workflows/old/autoprompt_ltxv_i2v.json` | other | ltx_legacy | i2v | retire-or-review |
| `workflows/old/autoprompt_relay_ltxv_i2v.json` | other | ltx_legacy | i2v | retire-or-review |
| `workflows/old/video_default_i2v_ltxv_msr_1actor_1background_v1.json` | video | ltx_legacy | i2v | retire-or-review |
| `workflows/old/video_default_i2v_ltxv_msr_1actor_1background_v2.json` | video | ltx_legacy | i2v | retire-or-review |
| `workflows/old/video_default_i2v_ltxv_msr_1actor_1background_v3.json` | video | ltx_legacy | i2v | retire-or-review |
| `workflows/old/video_default_ltxv_msr_1actor_1background_v1.json` | video | ltx_legacy | msr | retire-or-review |
| `workflows/old/video_default_ltxv_msr_1actor_1background_v2.json` | video | ltx_legacy | msr | retire-or-review |
| `workflows/old/video_default_ltxv_msr_1actor_1background_v3.json` | video | ltx_legacy | msr | retire-or-review |
| `workflows/old/video_ltxv_i2v_native_audio_v1.json` | video | ltx_legacy | i2v | retire-or-review |
| `workflows/old/video_ltxv_i2v_v1.json` | video | ltx_legacy | i2v | retire-or-review |
| `workflows/old/video_ltxv_ingredients_2stage_gguf_v3.json` | video | ltx_legacy | ingredients | retire-or-review |
| `workflows/old/video_ltxv_ingredients_2stage_gguf_v4.json` | video | ltx_legacy | ingredients | retire-or-review |
| `workflows/old/video_ltxv_ingredients_2stage_gguf_v5.json` | video | ltx_legacy | ingredients | retire-or-review |
| `workflows/old/video_ltxv_ingredients_2stage_v1.json` | video | ltx_legacy | ingredients | retire-or-review |
| `workflows/old/video_ltxv_ingredients_2stage_v2.json` | video | ltx_legacy | ingredients | retire-or-review |
| `workflows/old/video_ltxv_ingredients_2stage_v3.json` | video | ltx_legacy | ingredients | retire-or-review |
| `workflows/old/video_ltxv_ingredients_2stage_v4.json` | video | ltx_legacy | ingredients | retire-or-review |
| `workflows/old/video_ltxv_ingredients_2stage_v5.json` | video | ltx_legacy | ingredients | retire-or-review |
| `workflows/old/video_ltxv_ingredients_audio_2stage_gguf_v3.json` | video | ltx_legacy | ingredients | retire-or-review |
| `workflows/old/video_ltxv_ingredients_audio_2stage_gguf_v4.json` | video | ltx_legacy | ingredients | retire-or-review |
| `workflows/old/video_ltxv_ingredients_audio_2stage_gguf_v5.json` | video | ltx_legacy | ingredients | retire-or-review |
| `workflows/old/video_ltxv_ingredients_audio_2stage_v1.json` | video | ltx_legacy | ingredients | retire-or-review |
| `workflows/old/video_ltxv_ingredients_audio_2stage_v2.json` | video | ltx_legacy | ingredients | retire-or-review |
| `workflows/old/video_ltxv_ingredients_audio_2stage_v3.json` | video | ltx_legacy | ingredients | retire-or-review |
| `workflows/old/video_ltxv_ingredients_audio_2stage_v4.json` | video | ltx_legacy | ingredients | retire-or-review |
| `workflows/old/video_ltxv_ingredients_audio_2stage_v5.json` | video | ltx_legacy | ingredients | retire-or-review |
| `workflows/old/video_ltxv_ingredients_audio_v1.json` | video | ltx_legacy | ingredients | retire-or-review |
| `workflows/old/video_ltxv_ingredients_v1.json` | video | ltx_legacy | ingredients | retire-or-review |
| `workflows/old/video_ltxv_msr_1actor_1background_v1.json` | video | ltx_legacy | msr | retire-or-review |
| `workflows/old/video_ltxv_msr_1actor_1background_v2.json` | video | ltx_legacy | msr | retire-or-review |
| `workflows/old/video_ltxv_msr_1actor_1background_v3.json` | video | ltx_legacy | msr | retire-or-review |
| `workflows/sequence_to_sheet_minimax_h3_i2va_v1.json` | sequence | minimax_h3 | i2v | workflows/sequence/minimax_h3/ |
| `workflows/video_default_i2v_ltxv_msr_1actor_1background_v4.json` | video | ltx_legacy | i2v | workflows/video/ltx_legacy/ |
| `workflows/video_default_ltxv_msr_1actor_1background_v4.json` | video | ltx_legacy | msr | workflows/video/ltx_legacy/ |
| `workflows/video_ltxv_facefix_v1.json` | video | ltx_legacy | facefix | workflows/video/ltx_legacy/ |
| `workflows/video_ltxv_i2v_native_audio_v2.json` | video | ltx_legacy | i2v | workflows/video/ltx_legacy/ |
| `workflows/video_ltxv_i2v_v2.json` | video | ltx_legacy | i2v | workflows/video/ltx_legacy/ |
| `workflows/video_ltxv_ingredients_2stage_gguf_v6.json` | video | ltx_legacy | ingredients | workflows/video/ltx_legacy/ |
| `workflows/video_ltxv_ingredients_2stage_v6.json` | video | ltx_legacy | ingredients | workflows/video/ltx_legacy/ |
| `workflows/video_ltxv_ingredients_audio_2stage_gguf_v6.json` | video | ltx_legacy | ingredients | workflows/video/ltx_legacy/ |
| `workflows/video_ltxv_ingredients_audio_2stage_v6.json` | video | ltx_legacy | ingredients | workflows/video/ltx_legacy/ |
| `workflows/video_ltxv_msr_1actor_1background_v4.json` | video | ltx_legacy | msr | workflows/video/ltx_legacy/ |
| `workflows/video/minimax_h3/r2v_audio_eb57_8s_v1.json` | video | minimax_h3 | r2v | workflows/video/minimax_h3/ |
| `workflows/video/minimax_h3/r2v_audio_v1.json` | video | minimax_h3 | r2v | workflows/video/minimax_h3/ |
| `workflows/video/minimax_h3/r2v_audio_two_pass.json` | video | minimax_h3 | r2v | workflows/video/minimax_h3/ |
| `workflows/video/minimax_h3/r2v_audio_two_pass.profile.json` | video | minimax_h3 | profile | workflows/video/minimax_h3/ |
| `workflows/video/minimax_h3/r2v_eb57_8s_v1.json` | video | minimax_h3 | r2v | workflows/video/minimax_h3/ |
| `workflows/video/minimax_h3/r2v_turbo_8s_v1.json` | video | minimax_h3 | r2v | workflows/video/minimax_h3/ |
| `workflows/video/minimax_h3/r2v_v1.json` | video | minimax_h3 | r2v | workflows/video/minimax_h3/ |
| `workflows/video/minimax_h3/r2v_two_pass.json` | video | minimax_h3 | r2v | workflows/video/minimax_h3/ |
| `workflows/video/minimax_h3/r2v_two_pass.profile.json` | video | minimax_h3 | profile | workflows/video/minimax_h3/ |
| `workflows/video/minimax_h3/t2v.json` | video | minimax_h3 | t2v | workflows/video/minimax_h3/ |
| `workflows/video/minimax_h3/t2v_two_pass.json` | video | minimax_h3 | t2v | workflows/video/minimax_h3/ |
| `workflows/video/minimax_h3/t2v_two_pass.profile.json` | video | minimax_h3 | profile | workflows/video/minimax_h3/ |
| `workflows/video/ltx_25/capabilities.json` | video | ltx_2.5 | capability manifest | workflows/video/ltx_25/ |
| `workflows/video/ltx_25/profile-matrix.json` | video | ltx_2.5 | profile matrix | workflows/video/ltx_25/ |
| `workflows/video/ltx_25/t2v/t2v_draft.json` | video | ltx_2.5 | t2v | workflows/video/ltx_25/t2v/ |
| `workflows/video/ltx_25/t2v/t2v_draft.profile.json` | video | ltx_2.5 | profile | workflows/video/ltx_25/t2v/ |
| `workflows/video/ltx_25/t2v/t2v_standard.json` | video | ltx_2.5 | t2v | workflows/video/ltx_25/t2v/ |
| `workflows/video/ltx_25/t2v/t2v_standard.profile.json` | video | ltx_2.5 | profile | workflows/video/ltx_25/t2v/ |
| `workflows/video/ltx_25/t2v/t2v_final.json` | video | ltx_2.5 | t2v | workflows/video/ltx_25/t2v/ |
| `workflows/video/ltx_25/t2v/t2v_final.profile.json` | video | ltx_2.5 | profile | workflows/video/ltx_25/t2v/ |
| `workflows/video_seedvr2_3b_api.json` | video | seedvr | other | workflows/video/seedvr/ |

## Migration rules

- Treat this table as the complete tracked-JSON inventory; a CI check should compare it with `rg --files workflows -g '*.json'`.
- Resolve maintained callers through profile metadata or a centralized alias map; do not reconstruct paths from filenames at call sites.
- Keep old LTX assets in the retirement set until the LTX 2.5 cutover explicitly replaces them.
- Validate every moved JSON as an object and preserve its anchor/title contract before changing callers.
- Update this map in the same change as any workflow add, move, or retirement.



