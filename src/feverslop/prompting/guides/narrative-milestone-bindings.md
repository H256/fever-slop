Place only the requested missing narrative milestones in the story's canonical
locations. Return a JSON object with one key, "bindings", containing exactly
one object for each id in missing_milestone_ids, in that same order. Each
object has "milestone_id", "location_id", and "relative_position" (a number
from 0.0 to 1.0 within that location's story phase).

Use only ids from location_order and missing_milestone_ids. Read the full
story_idea and milestone_order to choose where each event occurs. Preserve
the narrative order: do not place a later event in an earlier location. Within
one location, use increasing relative_position values for successive events.
Do not invent locations, milestones, characters, or story facts. Return only
the compact JSON object; do not explain your choices.
