import unittest

from feverslop.domain.canonical_render_plan import PromptRole, build_canonical_scene
from feverslop.domain.effective_render_plan import project_effective_scene
from feverslop.domain.locked_scene_facts import LockedSceneFacts, locked_scene_facts_from_scene


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

    def test_projects_effective_references_and_directives_with_stable_provenance(self):
        canonical = build_canonical_scene(
            segment_id="segment-a",
            generated_roles={PromptRole.PERFORMANCE_TIMING: {"intent": "instrumental"}},
        )
        canonical["roles"][PromptRole.PERFORMANCE_TIMING]["override"] = {
            "value": {"intent": "singing"},
        }
        source = {
            "scene": 1,
            "canonical": canonical,
            "references": {
                "actor_ids": ["hero"],
                "actor_reference_descriptions": [{"id": "hero", "description": "silver coat"}],
                "location_id": "rooftop",
                "prop_bindings": {"hero": ["lantern"]},
                "audio_subject_bindings": {"vocals": {"subject_id": "hero", "speaker_id": "S1"}},
            },
            "subject_directives": {
                "schema_version": "subject-directives/v1",
                "shot_id": "segment-a",
                "temporal_scope": {"start_seconds": 0, "end_seconds": 4},
                "subjects": [{
                    "subject_id": "hero", "role": "singer", "position": "center",
                    "action": "raises lantern", "prop_bindings": [{"prop_id": "lantern", "state": "held"}],
                    "temporal_scope": {"start_seconds": 0, "end_seconds": 4},
                }],
                "spatial_relations": [],
            },
        }
        projected = project_effective_scene(source, canonical_scene=source)

        facts = locked_scene_facts_from_scene(projected)

        values = {(fact.category, fact.key): fact.value for fact in facts.facts}
        self.assertEqual('{"intent":"singing"}', values[("timing", "performance")])
        self.assertEqual("rooftop", values[("location", "id")])
        self.assertEqual('["lantern"]', values[("props", "hero")])
        self.assertEqual("raises lantern", values[("directive", "hero.action")])
        self.assertEqual("references:actor:hero", next(fact.source_id for fact in facts.facts if fact.key == "hero" and fact.category == "cast"))
        self.assertEqual("directive:segment-a:hero:action", next(
            fact.source_id for fact in facts.facts if fact.key == "hero.action"
        ))

    def test_rejects_legacy_contradictions_before_callers_can_use_facts(self):
        with self.assertRaisesRegex(ValueError, "legacy-a.*legacy-b"):
            locked_scene_facts_from_scene({
                "segment_id": "scene-01",
                "locked_facts": [
                    {"category": "wardrobe", "key": "hero", "value": "coat", "source_id": "legacy-a"},
                    {"category": "wardrobe", "key": "hero", "value": "jacket", "source_id": "legacy-b"},
                ],
            })


if __name__ == "__main__":
    unittest.main()
