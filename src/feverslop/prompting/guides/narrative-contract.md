Derive the structured narrative contract (the story arch) that the concept stage is
constrained by, from the supplied story idea, the canonical locations, and the
canonical actors. The supplied "locations" and "actors" arrays are the ONLY
canonical ids you may reference: every location id you emit must be one of the
supplied location ids, and every actor id you emit must be one of the supplied
actor ids. Never invent a new location or actor id; if the story does not need a
field, omit it (empty array or object) rather than filling it with invented
entries. Return ONLY valid JSON in this exact shape with these keys:
"location_order" (an ordered list of the canonical location ids the story visits,
each entry either an object {"id", "source"} or a plain string; "source" is a
short quote from the story idea that justifies the visit), "milestone_order"
(an ordered list of short snake_case narrative milestone ids that must occur in
this order, each entry either {"id", "source"} or a plain string),
"milestone_bindings" (one object per milestone: {"milestone_id", "location_id", "relative_position"}; use only canonical location ids and a relative_position from 0.0 to 1.0 within that location's phase),
"terminal_states" (an object keyed by canonical actor id, each value an object
{"milestone", "state", "reset_event", "source"} describing the actor's required
end state; omit actors whose end state does not matter),
"actor_allowed_locations" (an object keyed by canonical actor id, each value a list
of canonical location ids the actor is only allowed to appear at; omit actors with
no location restriction), "chronology_exceptions" (an object keyed by a short event
name, each value an object {"allows", "source"} where "allows" is a list of the
dimensions it relaxes, one of "milestone_order" or "location_order"; omit when no
exception is needed), and "one_shot_milestones" (a list of milestone ids from
milestone_order that must appear in exactly one scene; omit when none apply).
Keep the ordering faithful to the story idea: do not reorder, skip, or invent
beats the story does not state, and do not mark a milestone one-shot unless the
story idea makes it a single unrepeatable moment.
