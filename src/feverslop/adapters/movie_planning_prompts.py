from __future__ import annotations

import json
import re
from math import ceil

from feverslop.adapters.prompts import load_prompt_guide, load_template
from feverslop.domain.movie import CinematicShot, MovieActor, MovieBible, MovieLocation, StoryArch


def _story_arch_prompt(*, title: str, source_type: str, story_text: str, desired_length: float) -> str:
    template = load_template("story_arch_prompt")
    return template.render(
        title=title,
        source_type=source_type,
        story_text=story_text,
        desired_length=desired_length,
    ).strip()


def _shot_plan_prompt(*, story_arch: StoryArch, desired_length: float, width: int, height: int, min_duration: float, max_duration: float) -> str:
    target_shots = max(1, ceil(float(desired_length) / max(1.0, min(float(max_duration), 12.0))))
    template = load_template("shot_plan_prompt")
    return template.render(
        story_arch=story_arch,
        desired_length=desired_length,
        width=width,
        height=height,
        min_duration=f"{min_duration:g}",
        max_duration=f"{max_duration:g}",
        target_shots=target_shots,
        story_arch_json=json.dumps(list(story_arch.beats), ensure_ascii=False),
    ).strip()


def _movie_bible_prompt(*, title: str, source_type: str, story_text: str, desired_length: float, story_arch: StoryArch, config: dict) -> str:
    dialogue_language = _config_dialogue_language(config)
    dialogue_rule = f"- All dialogue in the movie bible and downstream shot plan must be in {dialogue_language} only." if dialogue_language else ""
    template = load_template("movie_bible_prompt")
    return template.render(
        title=title,
        source_type=source_type,
        story_text=story_text,
        desired_length=desired_length,
        dialogue_language=dialogue_language or "unspecified",
        dialogue_rule=dialogue_rule,
        story_arch_json=json.dumps({"title": story_arch.title, "premise": story_arch.premise, "beats": list(story_arch.beats)}, ensure_ascii=False),
        config_json=json.dumps(config, ensure_ascii=False),
    ).strip()


def _refine_location_prompts_prompt(locations: tuple[MovieLocation, ...], source_text: str, *, guide: str = "") -> str:
    loc_data = [
        {"id": loc.id, "name": loc.name, "visual_description": loc.visual_description}
        for loc in locations
    ]
    template = load_template("refine_location_prompts_prompt")
    return template.render(
        locations_json=json.dumps(loc_data, ensure_ascii=False),
        source_text=source_text,
        guide=guide,
    ).strip()


def _sanitize_location_image_prompt(value: str) -> str:
    text = re.sub(r"(?i)\b(?:cinematic\s+)?environment reference sheet(?:\s+for)?\b", "", value)
    text = re.sub(r"(?i)\breference sheet\b", "", text)
    return " ".join(text.split()).strip(" .,-")


def _refine_actor_prompts_prompt(actors: tuple[MovieActor, ...], source_text: str, premise: str, *, guide: str = "") -> str:
    actor_data = [
        {"id": actor.id, "name": actor.name, "role": actor.role, "visual_description": actor.visual_description}
        for actor in actors
    ]
    template = load_template("refine_actor_prompts_prompt")
    return template.render(
        actors_json=json.dumps(actor_data, ensure_ascii=False),
        source_text=source_text,
        premise=premise,
        guide=guide,
    ).strip()


def _krea_reference_guides(reference_hero_workflow: str | None) -> tuple[str, str]:
    """Return Krea guides only for a Krea reference-image workflow."""
    workflow = str(reference_hero_workflow or "").casefold()
    if "krea" not in workflow:
        return "", ""
    return load_prompt_guide("krea-location"), load_prompt_guide("krea-actor")


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
            screenplay_json = json.dumps(scenes_data, ensure_ascii=False)
    bible_json = json.dumps({
        "title": bible.title,
        "premise": bible.premise,
        "beats": list(bible.story_arch.beats),
        "actors": [asdict_like_actor(actor) for actor in bible.actors],
        "locations": [asdict_like_location(location) for location in bible.locations],
        "continuity": [rule.description for rule in bible.continuity],
        "style_constraints": list(bible.style_constraints),
    }, ensure_ascii=False)
    template = load_template("shot_plan_from_bible_prompt")
    return template.render(
        bible_json=bible_json,
        screenplay_json=screenplay_json,
        desired_length=desired_length,
        dialogue_language=dialogue_language or "unspecified",
        dialogue_language_or_default=dialogue_language or "the requested spoken dialogue language",
        dialogue_rule=dialogue_rule,
        width=width,
        height=height,
        min_duration=f"{min_duration:g}",
        max_duration=f"{max_duration:g}",
        target_shots=target_shots,
    ).strip()


def _movie_continuity_plan_prompt(*, title: str, source_type: str, story_text: str, desired_length: float, bible: MovieBible, shots: tuple[CinematicShot, ...], config: dict) -> str:
    dialogue_language = str((bible.runtime_constraints or {}).get("dialogue_language") or "").strip()
    screenplay_rule = "- Source is a screenplay: preserve scene order, dialogue cues, and character cues. Do not rewrite the screenplay structure." if source_type == "screenplay" else "- Source is an idea/short story: create a stronger causal chain while preserving the premise and constraints."
    max_scene_actors = int((bible.runtime_constraints or {}).get("max_scene_actors") or 4)
    bible_json = json.dumps({
        "title": bible.title,
        "premise": bible.premise,
        "actors": [asdict_like_actor(actor) for actor in bible.actors],
        "locations": [asdict_like_location(location) for location in bible.locations],
        "continuity": [rule.description for rule in bible.continuity],
        "style_constraints": list(bible.style_constraints),
    }, ensure_ascii=False)
    shots_json = json.dumps([{
        "shot_id": shot.shot_id,
        "description": shot.description,
        "action": shot.action,
        "camera": shot.camera,
        "acting": shot.expression,
        "dialogue": shot.dialogue,
        "actor_ids": list(shot.actor_ids),
        "location_id": shot.location_id,
        "location": shot.location,
    } for shot in shots], ensure_ascii=False)
    template = load_template("movie_continuity_plan_prompt")
    return template.render(
        title=title,
        source_type=source_type,
        story_text=story_text,
        desired_length=desired_length,
        config_json=json.dumps(config, ensure_ascii=False),
        bible_json=bible_json,
        shots_json=shots_json,
        dialogue_language=dialogue_language or "unspecified",
        dialogue_language_or_default=dialogue_language or "the requested spoken dialogue language",
        screenplay_rule=screenplay_rule,
        max_scene_actors=max_scene_actors,
    ).strip()


def _movie_story_design_prompt(*, title: str, source_type: str, story_text: str, desired_length: float, bible: MovieBible, story_arch: StoryArch, config: dict) -> str:
    dialogue_language = str((bible.runtime_constraints or {}).get("dialogue_language") or config.get("dialogue_language") or "").strip()
    max_scene_actors = int(config.get("max_scene_actors") or (bible.runtime_constraints or {}).get("max_scene_actors") or 4)
    screenplay_rule = "Analyze the supplied screenplay without rewriting its order or dialogue." if source_type == "screenplay" else "Design a strong short-film screenplay from the idea before any scene text is written."
    actor_ids = json.dumps([actor.id for actor in bible.actors], ensure_ascii=False)
    location_ids = json.dumps([location.id for location in bible.locations], ensure_ascii=False)
    story_arch_json = json.dumps({"title": story_arch.title, "premise": story_arch.premise, "beats": list(story_arch.beats)}, ensure_ascii=False)
    bible_json = json.dumps({"actors": [asdict_like_actor(actor) for actor in bible.actors], "locations": [asdict_like_location(location) for location in bible.locations]}, ensure_ascii=False)
    template = load_template("movie_story_design_prompt")
    return template.render(
        title=title,
        source_type=source_type,
        story_text=story_text,
        desired_length=desired_length,
        dialogue_language=dialogue_language or "unspecified",
        screenplay_rule=screenplay_rule,
        actor_ids=actor_ids,
        location_ids=location_ids,
        max_scene_actors=max_scene_actors,
        story_arch_json=story_arch_json,
        bible_json=bible_json,
        config_json=json.dumps(config, ensure_ascii=False),
    ).strip()


def _movie_screenplay_prompt(*, title: str, source_type: str, story_text: str, desired_length: float, bible: MovieBible, story_arch: StoryArch, story_design, config: dict) -> str:
    dialogue_language = str((bible.runtime_constraints or {}).get("dialogue_language") or config.get("dialogue_language") or "").strip()
    screenplay_rule = "Preserve source scene order and dialogue exactly; annotate it into structured scenes. Do not polish or rewrite the supplied screenplay." if source_type == "screenplay" else "Write an actual compact screenplay from STORY DESIGN, not just a scene list, without exceeding the target duration."
    design = movie_story_design_like(story_design)
    actor_ids = json.dumps([actor.id for actor in bible.actors], ensure_ascii=False)
    location_ids = json.dumps([location.id for location in bible.locations], ensure_ascii=False)
    story_arch_json = json.dumps({"title": story_arch.title, "premise": story_arch.premise, "beats": list(story_arch.beats)}, ensure_ascii=False)
    story_design_json = json.dumps(design, ensure_ascii=False)
    bible_json = json.dumps({"actors": [asdict_like_actor(actor) for actor in bible.actors], "locations": [asdict_like_location(location) for location in bible.locations]}, ensure_ascii=False)
    template = load_template("movie_screenplay_prompt")
    return template.render(
        title=title,
        source_type=source_type,
        story_text=story_text,
        desired_length=desired_length,
        dialogue_language=dialogue_language or "unspecified",
        dialogue_language_or_default=dialogue_language or "the requested spoken dialogue language",
        screenplay_rule=screenplay_rule,
        actor_ids=actor_ids,
        location_ids=location_ids,
        story_arch_json=story_arch_json,
        story_design_json=story_design_json,
        bible_json=bible_json,
        config_json=json.dumps(config, ensure_ascii=False),
    ).strip()


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
    template = load_template("movie_narrative_plan_prompt")
    return template.render(
        title=title,
        source_type=source_type,
        desired_length=desired_length,
        actor_ids=json.dumps([actor.id for actor in bible.actors], ensure_ascii=False),
        location_ids=json.dumps([location.id for location in bible.locations], ensure_ascii=False),
        config_json=json.dumps(config, ensure_ascii=False),
        screenplay_json=json.dumps(scenes, ensure_ascii=False),
    ).strip()


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
