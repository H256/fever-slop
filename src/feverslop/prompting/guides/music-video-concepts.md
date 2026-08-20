Create exactly one concise visual concept and scene reference assignment per supplied timed segment. Return only a JSON object whose keys exactly match segment_id. Treat the segments as one continuous visual story; preserve stable character/outfit/environment/lighting/motifs. Each concept must stand alone. Use visible actions, setting, lighting, composition, and emotion; avoid metaphors, abstract language, technical render parameters, quoted lyrics, new characters, and new locations. Honor location constraints, actor ids, structured location ids, subject mode, max scene actors, and prompt guidance. Name every references.actor_ids entry in concept, name every selected actor, and describe a visible coherent shared action and spatial relationship. A collective noun does not replace naming them. Return each value as {"concept": "...", "references": {"actor_ids": ["..."], "location_id": "location_id"}} when structured context is available; retain a plain concept string for legacy compatibility.
Legacy compatibility wording: location_constraint is mandatory; every scene concept must stand alone.
Continuity rules: Read all available segments first and infer the full emotional and visual arc. Maintain character, outfit, environment, lighting direction, motifs, and story progression. Repeat key visible continuity details in every standalone concept because the video model has no memory. Never write "the same character", "still", "continues", "next", "after", or "from earlier".
Do not invent or assume character details such as hair color, skin tone, age, ethnicity, eye color, or body type unless explicitly provided in the subject or context.
Prompt guidance categories include shot types, character visibility, environments, lighting, camera motion, physical interaction, facial expression, outfit rules, prompt structure, list handling, and word count.
For instrumental segments, advance the visual story without sung words and do not say the character is singing.

SEGMENT PERFORMANCE TYPE IS AUTHORITATIVE.

For every supplied segment, obey its `type` exactly:
- `instrumental`: The Lead Singer or Vocalist may be visible, but MUST NOT sing. Do not describe singing, lip-sync, vocal delivery, lyric delivery, open-mouth vocal performance, or any mouth movement implying vocals. Use non-vocal visible actions instead, such as swaying, breathing, looking, gesturing, holding a microphone silently, or moving with the beat.
- `vocals`: Visible singing/lip-sync is allowed when appropriate to the supplied segment and lyrics.
- `mixed`: Do not describe continuous singing. Vocal performance is allowed only for the vocal portion represented by the supplied segment data; otherwise use non-vocal performance actions.

The actor role `Vocalist`, the name `Lead Singer`, a microphone, the music-video setting, story idea, previous concepts, or continuity NEVER override the supplied segment `type`.
Do not infer singing merely because a singer is visible.
