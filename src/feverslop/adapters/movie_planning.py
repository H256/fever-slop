"""Re-export module for backwards compatibility.

All movie planning logic has been split into focused modules.
Import from the new modules directly for clarity:
  - feverslop.adapters.movie_planning_helpers
  - feverslop.adapters.movie_planning_bible
  - feverslop.adapters.movie_planning_prompts
  - feverslop.adapters.movie_planning_llm
  - feverslop.adapters.movie_planning_deterministic
"""
from __future__ import annotations

# Bible construction
# Character action detection (used by tests)
from feverslop.adapters.movie_planning_bible import (
    _actors_from_data,
    _actors_from_story_arch,
    _clean_visual_description,
    _config_style_constraints,
    _configured_actors,
    _configured_locations,
    _continuity_from_data,
    _dialogue_actor_names,
    _display_name,
    _is_character_action,
    _location_base_collides,
    _location_base_name,
    _location_id_matches,
    _location_visual_description,
    _locations_from_data,
    _locations_from_story_arch,
    _merge_screenplay_references,
    _movie_bible_from_data,
    _parse_screenplay_beat,
    _screenplay_reference_arch,
    _split_beats,
    _split_screenplay_beats,
    _visual_location_action,
)
from feverslop.adapters.movie_planning_deterministic import DeterministicMoviePlanner

# Shared helpers
from feverslop.adapters.movie_planning_helpers import (
    _beat_text,
    _dialogue_actor_ids,
    _ensure_minimum_actors,
    _minimum_actor_count,
    _normalize_movie_shots,
    _safe_id,
    _shot_part_text,
    _shots_from_data,
    _string_list,
    _transition_from_previous,
)

# Planners
from feverslop.adapters.movie_planning_llm import LLMMoviePlanner

__all__ = [
    # Planners
    "LLMMoviePlanner",
    "DeterministicMoviePlanner",
    # Bible
    "_actors_from_data",
    "_actors_from_story_arch",
    "_clean_visual_description",
    "_config_style_constraints",
    "_configured_actors",
    "_configured_locations",
    "_continuity_from_data",
    "_display_name",
    "_dialogue_actor_names",
    "_location_base_collides",
    "_location_base_name",
    "_location_id_matches",
    "_location_visual_description",
    "_locations_from_data",
    "_locations_from_story_arch",
    "_merge_screenplay_references",
    "_movie_bible_from_data",
    "_parse_screenplay_beat",
    "_screenplay_reference_arch",
    "_split_beats",
    "_split_screenplay_beats",
    "_visual_location_action",
    "_is_character_action",
    # Helpers
    "_beat_text",
    "_dialogue_actor_ids",
    "_ensure_minimum_actors",
    "_minimum_actor_count",
    "_normalize_movie_shots",
    "_safe_id",
    "_shot_part_text",
    "_shots_from_data",
    "_string_list",
    "_transition_from_previous",
]
