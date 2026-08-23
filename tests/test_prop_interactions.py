import unittest

from feverslop.domain.reference_workspace import (
    PropInteraction,
    SceneReferenceAssignment,
)
from feverslop.prompting.scene_prompt_builder import normalize_scene_references


class PropInteractionTests(unittest.TestCase):
    def test_interaction_validates_actor_and_prop_ids_and_projects_semantics(self):
        interaction = PropInteraction(actor_id="ava", prop_id="guitar", action="holds", relationship="instrument")
        assignment = SceneReferenceAssignment(
            scene_number=1, actor_ids=("ava",), prop_ids=("guitar",),
            prop_interactions=(interaction,),
        )
        self.assertEqual([], assignment.validate_against(
            known_actor_ids=["ava"], known_location_ids=[], known_background_ids=[], known_prop_ids=["guitar"], max_scene_actors=4,
        ))
        self.assertEqual("holds", interaction.to_dict()["action"])
        self.assertEqual(["guitar"], normalize_scene_references(
            {"actor_ids": ["ava"], "prop_ids": ["guitar"], "prop_interactions": [interaction.to_dict()]},
            {"actors": [{"id": "ava"}], "structured_locations": [], "props": [{"id": "guitar"}]},
        )["prop_ids"])

    def test_unknown_prop_or_actor_is_rejected_but_legacy_assignment_is_unchanged(self):
        invalid = SceneReferenceAssignment(scene_number=1, actor_ids=("ava",), prop_ids=("guitar",), prop_interactions=(
            PropInteraction("ava", "guitar", "holds"),
        ))
        self.assertTrue(invalid.validate_against(
            known_actor_ids=["ava"], known_location_ids=[], known_background_ids=[], known_prop_ids=[], max_scene_actors=4,
        ))
        legacy = SceneReferenceAssignment(scene_number=1, actor_ids=("ava",))
        self.assertEqual((), legacy.prop_ids)
        self.assertEqual((), legacy.prop_interactions)


if __name__ == "__main__":
    unittest.main()
