Repair only missing music-video concept keys. Return valid JSON with exactly the keys in MISSING_SEGMENTS and one concise visual concept per key. Preserve continuity and location constraints, do not describe subject identity, outfit, or hair, and do not repeat full prompts.

SEGMENT PERFORMANCE TYPE IS AUTHORITATIVE.

For every item in `MISSING_SEGMENTS`, obey its `type` exactly:
- `instrumental`: The Lead Singer or Vocalist may be visible, but MUST NOT sing. Do not describe singing, lip-sync, vocal delivery, lyric delivery, open-mouth vocal performance, or mouth movement implying vocals. Use non-vocal performance actions instead.
- `vocals`: Visible singing/lip-sync is allowed when appropriate to the supplied lyrics and segment.
- `mixed`: Do not describe continuous singing; only represent vocal performance when supported by the supplied segment data.

Never infer singing from the actor role `Vocalist`, the name `Lead Singer`, a microphone, music-video context, previous concepts, or story continuity.
