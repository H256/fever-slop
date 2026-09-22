"""Focused tests for StoryPlanService._resolve_entities (L2).

Covers the three resolution decisions:
- use:    config entity with a non-empty description
- extend: config entity with an empty description (gets filled)
- invent: entity id not present in the config
"""

from __future__ import annotations

import unittest

from feverslop.application.story_plan_service import StoryPlanService

from .test_story_plan_dspy import FakePromptModules, make_request


def _service() -> StoryPlanService:
    return StoryPlanService(prompt_modules=FakePromptModules())


def _allocation(
    *,
    beat_character_ids: list[str] | None = None,
    beat_location_id: str | None = None,
    beat_prop_ids: list[str] | None = None,
    brief_character_ids: list[str] | None = None,
    brief_location_id: str | None = None,
    brief_prop_ids: list[str] | None = None,
) -> dict:
    return {
        "typed_beats": [
            {
                "character_ids": beat_character_ids or [],
                "location_id": beat_location_id,
                "prop_ids": beat_prop_ids or [],
            }
        ],
        "briefs": [
            {
                "brief_id": "brief-seg-1",
                "character_ids": brief_character_ids or [],
                "location_id": brief_location_id,
                "prop_ids": brief_prop_ids or [],
            }
        ],
    }


class EntityResolutionUseTests(unittest.TestCase):
    def test_config_entity_with_description_is_used_verbatim(self) -> None:
        request = make_request(
            characters=(
                {"id": "char-1", "name": "Singer", "description": "The lead."},
            ),
            locations=(
                {"id": "cave", "name": "Cave", "description": "A dark cave."},
            ),
            props=(
                {"id": "well", "name": "Well", "description": "An old well."},
            ),
        )
        resolved, decisions = _service()._resolve_entities(
            request,
            _allocation(
                beat_character_ids=["char-1"],
                beat_location_id="cave",
                beat_prop_ids=["well"],
            ),
        )
        by_type = {
            d["entity_type"]: d["decision"] for d in decisions
        }
        self.assertEqual(
            by_type,
            {"character": "use", "location": "use", "prop": "use"},
        )
        cave = next(e for e in resolved["locations"] if e["id"] == "cave")
        self.assertEqual(cave["description"], "A dark cave.")


class EntityResolutionExtendTests(unittest.TestCase):
    def test_config_entity_with_empty_description_is_extended(self) -> None:
        request = make_request(
            characters=(
                {"id": "char-1", "name": "Singer", "description": ""},
            ),
        )
        resolved, decisions = _service()._resolve_entities(
            request,
            _allocation(brief_character_ids=["char-1"]),
        )
        self.assertEqual(
            {d["entity_type"]: d["decision"] for d in decisions},
            {"character": "extend"},
        )
        singer = next(e for e in resolved["characters"] if e["id"] == "char-1")
        self.assertTrue(singer["description"])
        # The description is constrained to the story's creative direction.
        self.assertIn("Keep it intimate", singer["description"])

    def test_unreferenced_config_entities_are_kept_verbatim(self) -> None:
        request = make_request(
            characters=(
                {"id": "char-1", "name": "Singer", "description": "The lead."},
                {"id": "char-2", "name": "Backer", "description": "Backs the song."},
            ),
        )
        resolved, decisions = _service()._resolve_entities(
            request,
            _allocation(brief_character_ids=["char-1"]),
        )
        ids = {e["id"] for e in resolved["characters"]}
        self.assertEqual(ids, {"char-1", "char-2"})
        # Only the referenced entity is recorded as a decision.
        self.assertEqual([d["entity_id"] for d in decisions], ["char-1"])


class EntityResolutionInventTests(unittest.TestCase):
    def test_unknown_character_id_is_invented_with_derived_name(self) -> None:
        request = make_request(
            characters=(
                {"id": "char-1", "name": "Singer", "description": "The lead."},
            ),
        )
        resolved, decisions = _service()._resolve_entities(
            request,
            _allocation(brief_character_ids=["char-1", "ghost_woman"]),
        )
        by_id = {d["entity_id"]: d for d in decisions}
        self.assertEqual(by_id["char-1"]["decision"], "use")
        self.assertEqual(by_id["ghost_woman"]["decision"], "invent")
        self.assertEqual(by_id["ghost_woman"]["name"], "Ghost Woman")
        ghost = next(e for e in resolved["characters"] if e["id"] == "ghost_woman")
        self.assertEqual(ghost["name"], "Ghost Woman")
        self.assertTrue(ghost["description"])
        # Invented characters are never marked as the singer.
        self.assertFalse(ghost.get("is_singer", False))

    def test_unknown_location_and_prop_ids_are_invented(self) -> None:
        request = make_request()
        resolved, decisions = _service()._resolve_entities(
            request,
            _allocation(
                brief_location_id="haunted_mountain",
                brief_prop_ids=["old_well"],
            ),
        )
        by_type = {d["entity_type"]: d["decision"] for d in decisions}
        self.assertEqual(by_type, {"location": "invent", "prop": "invent"})
        self.assertTrue(any(e["id"] == "haunted_mountain" for e in resolved["locations"]))
        self.assertTrue(any(e["id"] == "old_well" for e in resolved["props"]))

    def test_invented_character_without_creative_direction_uses_song_title(self) -> None:
        request = make_request(source_evidence={})
        resolved, decisions = _service()._resolve_entities(
            request,
            _allocation(brief_character_ids=["new_face"]),
        )
        self.assertEqual(decisions[0]["decision"], "invent")
        face = next(e for e in resolved["characters"] if e["id"] == "new_face")
        self.assertIn("Demo Song", face["description"])
