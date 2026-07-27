from __future__ import annotations

import json
import re
from math import ceil

from feverslop.domain.movie import CinematicShot, MovieActor, MovieBible, MovieLocation, StoryArch


def _story_arch_prompt(*, title: str, source_type: str, story_text: str, desired_length: float) -> str:
    return f"""
Create a movie story arch from this {source_type}.
Title: {title}
Target duration seconds: {desired_length}

Return JSON with:
{{"title": string, "premise": string, "beats": [string]}}

Source:
{story_text}
""".strip()


def _shot_plan_prompt(*, story_arch: StoryArch, desired_length: float, width: int, height: int, min_duration: float, max_duration: float) -> str:
    target_shots = max(1, ceil(float(desired_length) / max(1.0, min(float(max_duration), 12.0))))
    return f"""
Create a continuous cinematic shot plan from this story arch.
Title: {story_arch.title}
Premise: {story_arch.premise}
Beats: {json.dumps(list(story_arch.beats), ensure_ascii=False)}
Target duration seconds: {desired_length}
Resolution: {width}x{height}
Target shot count: about {target_shots}. Prefer varied shot durations from {min_duration:g} to {max_duration:g} seconds. Never exceed {max_duration:g} seconds for one shot.

Return JSON with:
{{"shots": [{{"description": string, "duration_seconds": number, "camera": string, "action": string, "expression": string, "location": string, "dialogue": string, "actor_ids": [string], "location_id": string, "transition_from_previous": "cut|continuous"}}]}}

Rules:
- Write every non-dialogue prose field in English: description, camera, action, expression, location, and continuity-like text. Only the dialogue field may use the requested spoken dialogue language.
- If the source or steering names actors/characters, preserve them as stable snake_case actor_ids.
- If the idea asks for at least N characters, create at least N distinct actor_ids across the shot plan.
- Use stable snake_case location_id values for recurring locations.
- Use transition_from_previous="continuous" only when this shot directly continues the previous shot in the same location with overlapping actors, no time jump, no perspective jump, and no new story beat. Otherwise use "cut". First shot is always "cut".
""".strip()


def _movie_bible_prompt(*, title: str, source_type: str, story_text: str, desired_length: float, story_arch: StoryArch, config: dict) -> str:
    dialogue_language = _config_dialogue_language(config)
    dialogue_rule = f"- All dialogue in the movie bible and downstream shot plan must be in {dialogue_language} only." if dialogue_language else ""
    return f"""
Create a movie bible for this {source_type}.
Title: {title}
Target duration seconds: {desired_length}
Dialogue language: {dialogue_language or "unspecified"}
Story arch: {json.dumps({"title": story_arch.title, "premise": story_arch.premise, "beats": list(story_arch.beats)}, ensure_ascii=False)}
Config constraints: {json.dumps(config, ensure_ascii=False)}

Return JSON with:
{{"title": string, "premise": string, "actors": [{{"id": snake_case, "name": string, "role": string, "visual_description": string}}], "locations": [{{"id": snake_case, "name": string, "visual_description": string}}], "continuity": [{{"id": snake_case, "description": string}}], "style_constraints": [string]}}

Rules:
- Write all prose fields in English: premise, actor visual_description, location visual_description, continuity descriptions, and style_constraints. Actor/location names and ids may preserve source/config labels.
- If config.actors is present, use exactly those actor ids and do not invent actor ids.
- If config.structured_locations or config.locations is present, use exactly those location ids and do not invent location ids.
- If source_type is screenplay, derive actors from explicit screenplay character cues and locations from scene headings. Do not add a generic main_character when named cues exist.
- Actor and location visual_description must describe stable visual identity only, not camera moves, shots, dialogue, or reference-sheet layout.
- Never use placeholder phrases like "story-defined cinematic character", "story-defined cinematic location", "consistent face", "consistent body shape", or "consistent production design" as visual descriptions.
- If the source does not give enough appearance detail, use only the clean actor or location name as visual_description.
- Preserve screenplay dialogue cues in continuity/story structure, not in visual descriptions.
{dialogue_rule}

Source:
{story_text}
""".strip()


def _refine_location_prompts_prompt(locations: tuple[MovieLocation, ...], source_text: str) -> str:
    loc_data = [
        {"id": loc.id, "name": loc.name, "visual_description": loc.visual_description}
        for loc in locations
    ]
    return f"""
Refine the visual_description and image_prompt for each location below.

Current locations:
{json.dumps(loc_data, ensure_ascii=False)}

Source screenplay/story:
{source_text}

Return JSON with:
{{"locations": [{{"id": string, "visual_description": string, "image_prompt": string}}]}}

Rules for visual_description:
- Describe only the physical environment, production design, and atmosphere of the location.
- Remove all character names, character actions, dialogue, narrative prose, and camera directions.
- Remove screenplay heading syntax (all-caps labels, "INT./EXT.", time-of-day suffixes).
- If the current description is a bare word or heading (e.g. "GARDEN"), expand it into a descriptive, evocative environment prose sentence using the source text for context.
- Keep location-defining objects, textures, materials, and lighting (e.g. hearth, jars, trees, bark faces, stone, roots, fog, light quality).
- Write in English, one concise paragraph (up to 200 characters).

Rules for image_prompt:
- This prompt will be fed directly to an image generator to create one environment photograph.
- It must describe a wide establishing view of the location's production design, lighting, and atmosphere.
- It must end with: "Wide establishing view, production design, lighting, atmosphere, single continuous image, no collage, no split screen, no panels, no people, no text."
- Never use the phrases "environment reference sheet" or "reference sheet".
- No characters, no action, no narrative, no camera moves.
- Write in English.
""".strip()


def _sanitize_location_image_prompt(value: str) -> str:
    text = re.sub(r"(?i)\b(?:cinematic\s+)?environment reference sheet(?:\s+for)?\b", "", value)
    text = re.sub(r"(?i)\breference sheet\b", "", text)
    return " ".join(text.split()).strip(" .,-")


def _refine_actor_prompts_prompt(actors: tuple[MovieActor, ...], source_text: str, premise: str) -> str:
    actor_data = [
        {"id": actor.id, "name": actor.name, "role": actor.role, "visual_description": actor.visual_description}
        for actor in actors
    ]
    return f"""
Refine the visual_description and image_prompt for each actor below.

Premise: {premise}

Current actors:
{json.dumps(actor_data, ensure_ascii=False)}

Source screenplay/story:
{source_text}

Return JSON with:
{{"actors": [{{"id": string, "visual_description": string, "image_prompt": string}}]}}

Rules for visual_description:
- Describe only the stable physical appearance and clothing of the actor. This is a reference-sheet description, not a scene description.
- MUST include: ethnicity/race fitting the setting and character context, face shape and features, skin tone, hair color and hairstyle, body type/stature, approximate age.
- MUST include: specific clothing/uniform with era-appropriate details (e.g., field-grey M1916 Schützentaucher for WWI German soldier, not just "uniform").
- MUST include: distinguishing marks if present or implied (scars, freckles, glasses, beard/stubble, missing teeth, etc.).
- Remove all character actions, emotions, dialogue, narrative prose, camera directions, and scene context.
- Remove screenplay heading syntax and time-of-day suffixes.
- If the current description is vague (e.g., "mud covered soldier"), expand it into specific physical details using the source text and premise for context clues.
- Write in English, one concise paragraph (up to 250 characters).
- Never use placeholder phrases like "story-defined cinematic character", "consistent face", "consistent body shape".

Rules for image_prompt:
- This prompt will be fed to an image generator to create a 4-panel character reference sheet.
- It must describe: full body front view, three-quarter face, close-up face, and full body profile, all showing the same consistent character.
- It must end with: "Four vertical panels in one image, full body front view, three-quarter face, close-up face, full body profile, white seamless background, neutral expression, consistent face, consistent wardrobe."
- No action, no narrative, no camera moves, no other characters.
- Write in English.
""".strip()


def _shot_plan_from_bible_prompt(*, bible: MovieBible, screenplay, desired_length: float, width: int, height: int, min_duration: float, max_duration: float) -> str:
    target_shots = max(1, ceil(float(desired_length) / max(1.0, min(float(max_duration), 12.0))))
    dialogue_language = str((bible.runtime_constraints or {}).get("dialogue_language") or "").strip()
    dialogue_rule = f"- Every dialogue field must be written in {dialogue_language} only. Do not mix in any other spoken language." if dialogue_language else ""
    screenplay_json = ""
    if screenplay is not None:
        scenes_data = []
        for scene in getattr(screenplay, "scenes", []):
            scenes_data.append({
                "scene_id": scene.scene_id,
                "heading": scene.heading,
                "dialogue": scene.dialogue,
                "action": scene.action,
                "actor_ids": list(scene.actor_ids),
                "location_id": scene.location_id,
            })
        if scenes_data:
            screenplay_json = "\nSCREENPLAY: " + json.dumps(scenes_data, ensure_ascii=False)
    return f"""
Create a continuous cinematic render plan from this movie bible.
Bible: {json.dumps({
        "title": bible.title,
        "premise": bible.premise,
        "beats": list(bible.story_arch.beats),
        "actors": [asdict_like_actor(actor) for actor in bible.actors],
        "locations": [asdict_like_location(location) for location in bible.locations],
        "continuity": [rule.description for rule in bible.continuity],
        "style_constraints": list(bible.style_constraints),
    }, ensure_ascii=False)}
{screenplay_json}
Target duration seconds: {desired_length}
Dialogue language: {dialogue_language or "unspecified"}
Resolution: {width}x{height}
Target shot count: about {target_shots}. Prefer varied shot durations from {min_duration:g} to {max_duration:g} seconds. Never exceed {max_duration:g} seconds for one shot.

Return JSON with:
{{"shots": [{{"description": string, "duration_seconds": number, "camera": string, "action": string, "acting": string, "location": string, "dialogue": string, "actor_ids": [string], "location_id": string, "continuity_notes": string, "transition_from_previous": "cut|continuous"}}]}}

Rules:
- Write every non-dialogue prose field in English: description, camera, action, acting, location, and continuity_notes. Only dialogue may use {dialogue_language or "the requested spoken dialogue language"}.
- actor_ids must only use bible actor ids.
- Name every actor_ids entry in action and give each selected actor a visible contribution to the shot.
- For multiple actors, describe their spatial relationship and one coherent shared action.
- A collective noun such as party, group, or crowd does not replace naming each selected actor.
- location_id must only use bible location ids.
- Never put more than 4 actors in one shot.
- Preserve dialogue, camera, acting, action, and continuity as separate fields.
- If a voice comes from a radio, transmitter, speaker, recorder, future self, unseen source, or distorted entity, mark it with a clear dialogue cue such as "(Radio)" or "(Distorted Voice)" and describe the device/source in action. Do not write it as visible actor lipsync.
- Map screenplay scene dialogue to shots covering the same location and actors. Distribute scene dialogue across shots so the full scene dialogue appears across the shot sequence for that scene.
- Use transition_from_previous="continuous" only when this shot directly continues the previous shot in the same location with overlapping actors, no time jump, no perspective jump, and no new story beat. Otherwise use "cut". First shot is always "cut".
{dialogue_rule}
""".strip()


def _movie_continuity_plan_prompt(*, title: str, source_type: str, story_text: str, desired_length: float, bible: MovieBible, shots: tuple[CinematicShot, ...], config: dict) -> str:
    dialogue_language = str((bible.runtime_constraints or {}).get("dialogue_language") or "").strip()
    screenplay_rule = "- Source is a screenplay: preserve scene order, dialogue cues, and character cues. Do not rewrite the screenplay structure." if source_type == "screenplay" else "- Source is an idea/short story: create a stronger causal chain while preserving the premise and constraints."
    return f"""
Build a movie continuity plan for AI film generation.

Return valid JSON only. No markdown, no commentary.

Required top-level shape:
{{
  "continuity_ledger": {{
    "style_bible": {{"visual_style": "", "palette": "", "lighting": "", "camera": "", "negative_constraints": []}},
    "characters": {{"actor_id": {{"character_id": "", "base_identity": "", "wardrobe": "", "carried_props": [], "physical_state": "", "emotional_state": "", "last_location": "", "last_action": ""}}}},
    "locations": {{"location_id": {{"location_id": "", "name": "", "time_of_day": "", "lighting": "", "props": [], "environmental_state": ""}}}},
    "scene_order": []
  }},
  "scene_continuity": {{
    "shot_id": {{"shot_id": "", "location_id": "", "incoming": [], "required_carryovers": [], "allowed_changes": [], "outgoing": [], "characters": {{}}, "location": {{}}}}
  }},
  "narrative_chain": [
    {{"shot_id": "", "story_state_before": "", "story_state_after": "", "cause_from_previous": "", "narrative_purpose": "", "conflict_or_tension": "", "turning_point": "", "sets_up_next": ""}}
  ]
}}

Rules:
- Write all continuity, state, style, location, and narrative prose in English. Only quoted dialogue text may remain in {dialogue_language or "the requested spoken dialogue language"}.
- Keep actor ids exactly to bible actor ids. Keep location ids exactly to bible location ids.
- Every shot id from SHOTS must appear once in scene_order, scene_continuity, and narrative_chain.
- Every shot after the first needs a concrete cause_from_previous.
- Every shot except the last needs a concrete sets_up_next.
- required_carryovers are facts the generator must preserve in this shot.
- allowed_changes are facts allowed or expected to change in this shot.
- outgoing states become useful incoming state for later shots.
- Never invent more than {int((bible.runtime_constraints or {}).get("max_scene_actors") or 4)} visible actors in one shot.
- Dialogue language is {dialogue_language or "unspecified"}; any dialogue continuity must respect it.
{screenplay_rule}

TITLE: {title}
TARGET DURATION: {desired_length}
CONFIG CONSTRAINTS: {json.dumps(config, ensure_ascii=False)}
BIBLE: {json.dumps({
        "title": bible.title,
        "premise": bible.premise,
        "actors": [asdict_like_actor(actor) for actor in bible.actors],
        "locations": [asdict_like_location(location) for location in bible.locations],
        "continuity": [rule.description for rule in bible.continuity],
        "style_constraints": list(bible.style_constraints),
    }, ensure_ascii=False)}
SHOTS: {json.dumps([{
        "shot_id": shot.shot_id,
        "description": shot.description,
        "action": shot.action,
        "camera": shot.camera,
        "acting": shot.expression,
        "dialogue": shot.dialogue,
        "actor_ids": list(shot.actor_ids),
        "location_id": shot.location_id,
        "location": shot.location,
    } for shot in shots], ensure_ascii=False)}
SOURCE:
{story_text}
""".strip()


def _movie_story_design_prompt(*, title: str, source_type: str, story_text: str, desired_length: float, bible: MovieBible, story_arch: StoryArch, config: dict) -> str:
    dialogue_language = str((bible.runtime_constraints or {}).get("dialogue_language") or config.get("dialogue_language") or "").strip()
    max_scene_actors = int(config.get("max_scene_actors") or (bible.runtime_constraints or {}).get("max_scene_actors") or 4)
    screenplay_rule = "Analyze the supplied screenplay without rewriting its order or dialogue." if source_type == "screenplay" else "Design a strong short-film screenplay from the idea before any scene text is written."
    return f"""
Create the dramaturgical story design for this movie. This is pre-screenplay story editing, not renderer prompt writing.

Return JSON with:
{{"title": string, "premise": string, "theme": string, "act_structure": [{{"act_id": string, "title": string, "purpose": string, "scene_ids": [string]}}], "turning_points": [{{"id": string, "scene_id": string, "description": string}}], "setup_payoff_threads": [{{"id": string, "setup_scene_id": string, "payoff_scene_id": string, "description": string}}], "character_arcs": [{{"actor_id": string, "want": string, "need": string, "starting_state": string, "ending_state": string}}], "scene_blueprint": [{{"scene_id": "scene_0001", "purpose": string, "conflict": string, "emotional_turn": string, "subtext": string, "dialogue_function": string, "required_actors": [string], "location_id": string, "expected_duration": number}}]}}

Rules:
- Write all story-design prose in English: premise, theme, act purposes, turning points, setup/payoff descriptions, character arcs, scene purpose, conflict, emotional_turn, subtext, and dialogue_function.
- {screenplay_rule}
- Write screenplay craft, not a scene list: every scene needs a dramatic purpose, conflict, emotional turn, subtext, and dialogue function.
- actor_ids/required_actors must only use these ids: {[actor.id for actor in bible.actors]}
- location_id must only use these ids: {[location.id for location in bible.locations]}
- No scene may require more than {max_scene_actors} actors.
- Dialogue language is {dialogue_language or "unspecified"}.
- Do not include camera, renderer, ComfyUI, MSR, reference-sheet, or visual prompt instructions.
- Target total duration seconds: {desired_length}.

Title: {title}
Story arch: {json.dumps({"title": story_arch.title, "premise": story_arch.premise, "beats": list(story_arch.beats)}, ensure_ascii=False)}
Bible: {json.dumps({"actors": [asdict_like_actor(actor) for actor in bible.actors], "locations": [asdict_like_location(location) for location in bible.locations]}, ensure_ascii=False)}
Config: {json.dumps(config, ensure_ascii=False)}
Source:
{story_text}
""".strip()


def _movie_screenplay_prompt(*, title: str, source_type: str, story_text: str, desired_length: float, bible: MovieBible, story_arch: StoryArch, story_design, config: dict) -> str:
    dialogue_language = str((bible.runtime_constraints or {}).get("dialogue_language") or config.get("dialogue_language") or "").strip()
    screenplay_rule = "Preserve source scene order and dialogue exactly; annotate it into structured scenes. Do not polish or rewrite the supplied screenplay." if source_type == "screenplay" else "Write an actual compact screenplay from STORY DESIGN, not just a scene list, without exceeding the target duration."
    design = movie_story_design_like(story_design)
    return f"""
Create the canonical structured screenplay for this movie.

Return JSON with:
{{"title": string, "source_type": "{source_type}", "dialogue_language": string, "scenes": [{{"scene_id": "scene_0001", "heading": string, "summary": string, "action": string, "dialogue": string, "actor_ids": [string], "location_id": string, "source_span": string, "dramatic_purpose": string, "conflict": string, "emotional_turn": string, "subtext": string, "dialogue_function": string}}]}}

Rules:
- Write every non-dialogue screenplay field in English: heading, summary, action, source_span, dramatic_purpose, conflict, emotional_turn, subtext, and dialogue_function.
- Write only the dialogue field in {dialogue_language or "the requested spoken dialogue language"}. Do not put translated dialogue in action or summary.
- {screenplay_rule}
- Every scene must visibly implement its matching STORY DESIGN scene_blueprint.
- Dialogue is mandatory. Every scene with two or more actors must include spoken dialogue where each present actor speaks at least once. Every scene with a single actor must include voiceover, narration, monologue, or phone/radio dialogue. Empty dialogue fields are not acceptable.
- For screenplay input, preserve source dialogue and order exactly, but fill dramaturgical annotation fields from STORY DESIGN.
- actor_ids must only use these ids: {[actor.id for actor in bible.actors]}
- Name every actor_ids entry in action and describe how each selected actor participates in the scene.
- For multiple actors, describe their spatial relationship and coherent shared action without rewriting supplied dialogue.
- A collective noun such as party, group, or crowd does not replace naming each selected actor.
- location_id must only use these ids: {[location.id for location in bible.locations]}
- Dialogue language is {dialogue_language or "unspecified"}.
- Keep screenplay text and dialogue out of actor/location visual descriptions.
- Do not include camera, renderer, ComfyUI, MSR, reference-sheet, or visual prompt instructions.

Title: {title}
Target duration seconds: {desired_length}
Story arch: {json.dumps({"title": story_arch.title, "premise": story_arch.premise, "beats": list(story_arch.beats)}, ensure_ascii=False)}
STORY DESIGN: {json.dumps(design, ensure_ascii=False)}
Bible: {json.dumps({"actors": [asdict_like_actor(actor) for actor in bible.actors], "locations": [asdict_like_location(location) for location in bible.locations]}, ensure_ascii=False)}
Config: {json.dumps(config, ensure_ascii=False)}
Source:
{story_text}
""".strip()


def _movie_narrative_plan_prompt(*, title: str, source_type: str, desired_length: float, bible: MovieBible, screenplay, config: dict) -> str:
    scenes = [
        {
            "scene_id": scene.scene_id,
            "summary": scene.summary,
            "action": scene.action,
            "dialogue": scene.dialogue,
            "actor_ids": list(scene.actor_ids),
            "location_id": scene.location_id,
        }
        for scene in getattr(screenplay, "scenes", ())
    ]
    return f"""
Create a narrative memory plan from the canonical screenplay.

Return JSON with:
{{"title": string, "sequences": [{{"sequence_id": string, "title": string, "scene_ids": [string], "dramatic_function": string}}], "causal_chain": [{{"scene_id": string, "story_state_before": string, "story_state_after": string, "cause_from_previous": string, "sets_up_next": string}}], "open_threads": [string]}}

Rules:
- Write all narrative memory prose in English. Preserve quoted dialogue only if needed and only in the screenplay dialogue language.
- Use only scene_id values from SCREENPLAY.
- Preserve scene order.
- Every scene after the first needs a concrete cause_from_previous.
- Every scene except the last needs a concrete sets_up_next.
- This is planning memory only; do not write renderer prompt prose.

Title: {title}
Source type: {source_type}
Target duration seconds: {desired_length}
Bible actors: {[actor.id for actor in bible.actors]}
Bible locations: {[location.id for location in bible.locations]}
Config: {json.dumps(config, ensure_ascii=False)}
SCREENPLAY: {json.dumps(scenes, ensure_ascii=False)}
""".strip()


def _config_dialogue_language(config: dict) -> str:
    return str(config.get("dialogue_language") or "").strip()


def asdict_like_actor(actor: MovieActor) -> dict:
    return {"id": actor.id, "name": actor.name, "role": actor.role, "visual_description": actor.visual_description}


def asdict_like_location(location: MovieLocation) -> dict:
    return {"id": location.id, "name": location.name, "visual_description": location.visual_description, "image_prompt": location.image_prompt}


def movie_story_design_like(story_design) -> dict:
    return {
        "title": getattr(story_design, "title", ""),
        "premise": getattr(story_design, "premise", ""),
        "theme": getattr(story_design, "theme", ""),
        "act_structure": [getattr(item, "__dict__", item) for item in getattr(story_design, "act_structure", ())],
        "turning_points": [getattr(item, "__dict__", item) for item in getattr(story_design, "turning_points", ())],
        "setup_payoff_threads": [getattr(item, "__dict__", item) for item in getattr(story_design, "setup_payoff_threads", ())],
        "character_arcs": [getattr(item, "__dict__", item) for item in getattr(story_design, "character_arcs", ())],
        "scene_blueprint": [getattr(item, "__dict__", item) for item in getattr(story_design, "scene_blueprint", ())],
    }
