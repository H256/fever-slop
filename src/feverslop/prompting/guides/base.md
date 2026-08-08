# Video Prompt Writing Guide (T2VA / I2VA / FL2VA / L2VA)

This is the authoritative base-video guide used by the integrated DSPy
renderer. The output has three fields, in this order:

```text
integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...
```

Write the multimodal description in playback order. `[Shot 1]` has no
timestamp; later shots use increasing cut times inside the video duration.
Describe visual style, composition, subjects, actions, camera movement,
dialogue, singing, and diegetic sound. Use natural camera vocabulary such as
`Push In`, `Pull Out`, `Pan`, `Truck`, `Tilt`, `Tracking Shot`, `Static Shot`,
and `POV`, adding amplitude and speed only when meaningful.

For I2V, begin with:
`For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.`
For FL2V, align the first and last pictures with the opening and ending time;
describe the continuous path between them. For L2V, align the last picture
with the effective final duration and describe a plausible path converging on
that frame.

Speaking subjects keep stable `(S1)`, `(S2)` identifiers. Put only the exact
spoken words inside `<d>[Language] ...</d>` and preserve user-provided
dialogue, lyrics, and visible text verbatim. Use `says in an off-screen
voiceover` and state that the on-screen lips remain closed. Use
`<scenetrans>` for dialogue crossing a cut and `<cutoff>` for speech cut off
by the video ending.

`overall_soundscape` is one paragraph of one to four English sentences about
ambience, physical action sounds, and non-verbal human sounds. Do not repeat
dialogue or singing there. `non_diegetic_music` describes only audience-only
music using instrumentation, tempo, and dynamics; use `N/A` when there is no
such music. Do not invent unsupported factual details when strict fidelity is
enabled.