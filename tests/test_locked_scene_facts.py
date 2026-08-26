import unittest

from feverslop.domain.locked_scene_facts import LockedSceneFacts


class LockedSceneFactsTests(unittest.TestCase):
    def test_normalizes_and_keeps_fact_provenance(self):
        facts = LockedSceneFacts.create(
            scene_id=" scene-02 ",
            facts=[
                {"category": "Wardrobe", "key": "hero", "value": "silver cloak", "source_id": "cast:hero", "provenance": "canonical"},
                {"category": "Location", "key": "primary", "value": "ruined gate", "source_id": "location:gate", "provenance": "override"},
            ],
        )

        self.assertEqual("scene-02", facts.scene_id)
        self.assertEqual(["location", "wardrobe"], [fact.category for fact in facts.facts])
        self.assertEqual("cast:hero", facts.facts[1].source_id)
        self.assertEqual(facts, LockedSceneFacts.from_dict(facts.to_dict()))

    def test_rejects_contradictory_values_with_source_ids(self):
        with self.assertRaisesRegex(ValueError, "cast:hero.*override:hero"):
            LockedSceneFacts.create(
                scene_id="scene-01",
                facts=[
                    {"category": "cast", "key": "hero", "value": "young", "source_id": "cast:hero"},
                    {"category": "cast", "key": "hero", "value": "old", "source_id": "override:hero"},
                ],
            )

    def test_rejects_missing_identity_fields(self):
        with self.assertRaises(ValueError):
            LockedSceneFacts.create(scene_id="", facts=[])
        with self.assertRaises(ValueError):
            LockedSceneFacts.create(scene_id="scene-01", facts=[{"category": "cast"}])


if __name__ == "__main__":
    unittest.main()
