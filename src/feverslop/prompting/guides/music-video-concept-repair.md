Repair missing or invalid music-video concept keys. Return valid JSON with exactly the keys in MISSING_SEGMENTS and INVALID_SEGMENTS and one concise visual concept per key. For invalid concepts, address every reason in INVALID_SEGMENTS. Preserve continuity and location constraints, do not describe subject identity, outfit, or hair, and do not repeat full prompts. Return structured values as {"concept": "...", "references": {"actor_ids": ["..."], "location_id": "..."}, "narrative": {"story_beat": "...", "objective": "...", "action": "...", "action_phase": "...", "milestones": ["..."], "location": "...", "cast_states": {"actor_id": "..."}, "props": {"prop_id": "..."}, "incoming": {"location": "...", "cast_states": {}, "props": {}, "action": "...", "action_phase": "..."}, "outgoing": {"location": "...", "cast_states": {}, "props": {}, "action": "...", "action_phase": "..."}, "transition_from_previous": "cut", "transition_events": [], "reset_events": [], "causal_events": []}}.

Each repair must advance at least one narrative dimension. Do not repeat a completed one-shot milestone or regress a monotonic prop state. A deliberate reprise or reset is valid only when the repaired scene names the exact repeated milestone or prop in "reset_events" as an explicit causal reset; unrelated reset events do not authorize it.

Make `incoming` compatible with the predecessor's `outgoing` state. Name an exact state change in `transition_events`; a hard cut is not an exception. Use `transition_from_previous: "continuous"` only when the action requires the predecessor boundary frame. Keep a terminally absent actor absent until the narrative contract's exact return event occurs; do not restore visual presence from an audio performer binding.

BOUNDARY_CONTEXT carries the exact accepted boundary states beside every key you repair, and those identifiers are the canonical vocabulary. State values are compared token-for-token: `lich_s_lair` and `center_of_lich_s_lair` are different locations, as are `gesturing_to_dragon` and `gesturing_toward_the_dragon`. In `incoming`, omit states inherited unchanged from the predecessor (their exact values are taken over automatically) or restate the predecessor's identifiers verbatim; name a state only when it genuinely changes, and list every such change in `transition_events`. Your `outgoing` must match the accepted successor's `incoming` identifiers exactly for every state that successor names.

Follow `milestone_order` and `location_order` from the narrative contract. A nonlinear rewind is valid only when the repair names an exact event in "causal_events" and that event's `chronology_exceptions` entry explicitly allows the rewound dimension. Never use a chronology exception to skip unresolved milestones forward.

SEGMENT PERFORMANCE TYPE IS AUTHORITATIVE.

For every item in `MISSING_SEGMENTS`, obey its `type` exactly:
- `instrumental`: The Lead Singer or Vocalist may be visible, but MUST NOT sing. Do not describe singing, lip-sync, vocal delivery, lyric delivery, open-mouth vocal performance, or mouth movement implying vocals. Use non-vocal performance actions instead.
- `vocals`: Visible singing/lip-sync is allowed when appropriate to the supplied lyrics and segment.
- `mixed`: Do not describe continuous singing; only represent vocal performance when supported by the supplied segment data.

Never infer singing from the actor role `Vocalist`, the name `Lead Singer`, a microphone, music-video context, previous concepts, or story continuity.
