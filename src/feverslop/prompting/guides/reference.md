# Full-Reference Mode Rewrite Output Format Guide

Write all six sections in this exact order and in English:

```text
subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:
```

Use only the exact labels supplied by the application: `<Subject N>`,
`<Picture N>`, `<Video N>`, and `<Audio N>`. A subject is reusable visible
content such as a person, object, environment, costume, or effect. A picture
is standalone when it is a first frame, keyframe, last frame, storyboard, or
composition anchor; otherwise cite it inside the subject definition. A video
describes an edited/continued source or its temporal structure. Audio is
numbered independently and describes copied or referenced audio behavior.

`subject_definitions` gives each tracked subject one line, including its
reference labels and important observable features. `summary` begins with a
task prefix such as `[reference generation + audio reference]` and uses only
previously supplied labels. `retention_analysis` has one line per referenced
label and uses the semantic modes `fully_preserved`, `partially_preserved`,
`attribute_transfer`, `style_transfer`, `environment_transfer`,
`motion_transfer`, `audio_transfer`, or `weak_reference` as appropriate; do
not use full preservation merely because a reference is visible.

`detailed_description` is the main shot-by-shot playback description. Establish
style before `[Shot 1]`, then include composition, subject position and
appearance, actions, state changes, lighting, camera movement, sound, and
reference labels where their roles apply. Use stable `(S1)`, `(S2)` speaker
IDs and `<d>[Language] ...</d>` for exact dialogue or lyrics. Keep the base
guide's shot, camera, continuity, and visible-text rules. Do not repeat full
dialogue or lyrics in the two sound sections.

`overall_soundscape` summarizes ambience and physical sounds. `non_diegetic_music`
describes only music audible to the audience. When audio is reused, state the
copy/reference relationship in the section matching that audible layer. Under
strict fidelity, the request, reference descriptions, analyses, and plan are
a closed factual world: do not invent subjects, props, expressions, events,
camera actions, dialogue, lyrics, or visible text.