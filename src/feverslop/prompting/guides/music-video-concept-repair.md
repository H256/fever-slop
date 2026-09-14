Repair missing or invalid music-video concept keys. Return valid JSON with exactly the keys in MISSING_SEGMENTS and INVALID_SEGMENTS and one concise visual concept per key. For invalid concepts, address every reason in INVALID_SEGMENTS. Preserve continuity and location constraints, do not describe subject identity, outfit, or hair, and do not repeat full prompts. Return structured values as {"concept": "...", "references": {"actor_ids": ["..."], "location_id": "..."}, "narrative": {"story_beat": "...", "objective": "...", "action": "...", "action_phase": "...", "milestones": ["..."], "location": "...", "cast_states": {"actor_id": "..."}, "props": {"prop_id": "..."}, "reset_events": [], "causal_events": []}}.

Each repair must advance at least one narrative dimension. Do not repeat a completed one-shot milestone or regress a monotonic prop state. A deliberate reprise or reset is valid only when the repaired scene names the exact repeated milestone or prop in "reset_events" as an explicit causal reset; unrelated reset events do not authorize it.

Follow `milestone_order` and `location_order` from the narrative contract. A nonlinear rewind is valid only when the repair names an exact event in "causal_events" and that event's `chronology_exceptions` entry explicitly allows the rewound dimension. Never use a chronology exception to skip unresolved milestones forward.

SEGMENT PERFORMANCE TYPE IS AUTHORITATIVE.

For every item in `MISSING_SEGMENTS`, obey its `type` exactly:
- `instrumental`: The Lead Singer or Vocalist may be visible, but MUST NOT sing. Do not describe singing, lip-sync, vocal delivery, lyric delivery, open-mouth vocal performance, or mouth movement implying vocals. Use non-vocal performance actions instead.
- `vocals`: Visible singing/lip-sync is allowed when appropriate to the supplied lyrics and segment.
- `mixed`: Do not describe continuous singing; only represent vocal performance when supported by the supplied segment data.

Never infer singing from the actor role `Vocalist`, the name `Lead Singer`, a microphone, music-video context, previous concepts, or story continuity.
