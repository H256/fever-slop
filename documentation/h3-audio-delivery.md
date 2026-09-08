# H3 audio sources

A reference slot, a generation guide and the final soundtrack have separate
roles. A file named `full_mix` in reference slot 2 does not make an audio guide
connected to slot 1 consume that file. The pipeline now resolves source paths,
content hashes, roles, node inputs and audio windows before prompt compilation.
The renderer checks the patched graph against that contract before submitting it.

To select a particular guide source, add `conditioning_source` to the selected
workflow's `.profile.json` sidecar:

```json
{
  "conditioning_source": "full_mix"
}
```

For a supported `MiniMaxH3AddGuide.audio` connection through
`TrimAudioDuration` to `LoadAudio`, the renderer wires the named source regardless
of its reference index. Without this field, `selected_performer` preserves the
existing numbered guide source. The node's display title is never proof of which
audio it consumes. An explicit source without a supported guide, or an unknown
audio transformation in its chain, produces an `h3_audio_*` error.

The resolved `h3_audio_sources` records travel with the prompt and render plan.
`reference` means an input reference; `conditioning` identifies the guide input;
`output_copy` identifies source audio connected directly to a supported
`CreateVideo` or `VHS_VideoCombine` output. Decoded generated audio does not prove
that an external reference was copied. Likewise, the graph alone does not prove
an `audience_score` role. The final music-video soundtrack assembly remains a
separate pipeline operation.

`music_intent=NONE` means no added audience-only score. It does not suppress an
existing copied instrumental source: that source is described in
`overall_soundscape`, while `non_diegetic_music` remains `N/A`. Source records also
override legacy filename-based copy assumptions. Workflow/profile fingerprints
and compiler revision 43 invalidate stale prompt checkpoints.

CPU tests verify reordered references, source identity, trim timing, direct output
copy, missing/contradictory topology, and checkpoint recompilation. No controlled
render comparison has established whether an additional full-mix reference harms
lip sync. This change does not remove that reference or change the default guide.
