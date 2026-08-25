import unittest
from unittest.mock import patch

from feverslop.domain.scene_cast import resolve_scene_cast, scene_cast_to_prompt_payload

ACTORS = [
    {"id": "warrior", "name": "Warrior"},
    {"id": "mage", "name": "Mage"},
    {"id": "rogue", "name": "Rogue"},
]


class SceneCastTests(unittest.TestCase):
    def test_multi_mode_preserves_selected_order(self):
        cast = resolve_scene_cast(
            selected_actor_ids=["warrior", "mage", "rogue"],
            available_actors=ACTORS,
            subject_mode="multi",
            max_scene_actors=4,
        )
        payload = scene_cast_to_prompt_payload(cast)
        self.assertEqual(["warrior", "mage", "rogue"], payload["visible_actor_ids"])
        self.assertEqual("warrior", payload["primary_actor_id"])
        self.assertTrue(payload["requires_group_staging"])

    def test_single_mode_keeps_only_first_selected_actor(self):
        cast = resolve_scene_cast(
            selected_actor_ids=["mage", "rogue"],
            available_actors=ACTORS,
            subject_mode="single",
            max_scene_actors=4,
        )
        self.assertEqual(("mage",), cast.visible_actor_ids)

    def test_unknown_actors_are_removed(self):
        cast = resolve_scene_cast(
            selected_actor_ids=["missing", "rogue"],
            available_actors=ACTORS,
            subject_mode="multi",
            max_scene_actors=4,
        )
        self.assertEqual(("rogue",), cast.visible_actor_ids)

    def test_location_only_does_not_fallback_to_first_actor(self):
        cast = resolve_scene_cast(
            selected_actor_ids=[],
            available_actors=ACTORS,
            subject_mode="location_only",
        )
        payload = scene_cast_to_prompt_payload(cast)
        self.assertEqual((), cast.visible_actor_ids)
        self.assertEqual("location_only", payload["subject_mode"])
        self.assertEqual("", payload["primary_actor_id"])

    def test_malformed_cast_fallback_warns_with_scene_and_reconstructed_id(self):
        with patch("feverslop.domain.scene_cast.warnings.warn") as warn:
            cast = resolve_scene_cast(
                selected_actor_ids=["missing"],
                available_actors=ACTORS,
                scene_number=7,
            )
        self.assertEqual(("warrior",), cast.visible_actor_ids)
        warn.assert_called_once()
        self.assertIn("Scene 7", warn.call_args.args[0])
        self.assertIn("warrior", warn.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
