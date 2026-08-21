import unittest

from feverslop.prompting.prompt_pipeline import MusicVideoPromptPipeline
from tests.prompt_fakes import MusicVideoModulesFake


class MusicVideoPromptPipelineTests(unittest.TestCase):
    def test_create_concept_prompts_includes_global_context_and_notes(self):
        modules = MusicVideoModulesFake(concepts={"segment_001": "A youth stands in the allowed forest."})
        pipeline = MusicVideoPromptPipeline(object(), prompt_modules=modules)

        pipeline.create_concept_prompts(
            stage1_segments=[{"segment_id": "segment_001", "type": "instrumental"}],
            story_idea="A forest myth.",
            global_context={
                "locations": ["ancient forest"],
                "location_constraint": "Allowed locations: ancient forest",
            },
            notes="Keep the spring visible.",
        )

        payload = modules.calls[0].payload

        self.assertEqual("Allowed locations: ancient forest", payload["GLOBAL_CONTEXT"]["location_constraint"])
        self.assertEqual("Keep the spring visible.", payload["NOTES"])
        self.assertIn("segment_001", payload["SEGMENT_TIMELINE_JSON"][0]["segment_id"])

    def test_subject_and_locations_prompt_requests_multi_actor_reference_data(self):
        modules = MusicVideoModulesFake(subject_locations={"subject": "a singer", "actors": [{"id": "singer", "name": "Mara"}]})
        pipeline = MusicVideoPromptPipeline(object(), prompt_modules=modules)

        result = pipeline.create_subject_and_locations("stage story")

        self.assertEqual("stage story", modules.calls[0].payload["story_idea"])
        self.assertEqual("singer", result["actors"][0]["id"])

    def test_location_image_prompt_avoids_and_removes_reference_sheet_wording(self):
        modules = MusicVideoModulesFake(subject_locations={"subject": "a singer", "actors": [], "locations": [{"id": "stage", "image_prompt": "Cinematic environment reference sheet for a dark stage"}]})
        result = MusicVideoPromptPipeline(object(), prompt_modules=modules).create_subject_and_locations("stage story")

        self.assertNotIn("reference sheet", result["locations"][0]["image_prompt"].lower())

    def test_create_final_scene_prompts_raises_for_missing_segments(self):
        """Missing segment IDs should produce a clear error, not a raw KeyError."""
        pipeline = MusicVideoPromptPipeline(object(), prompt_modules=MusicVideoModulesFake())

        with self.assertRaisesRegex(ValueError, "segment_002"):
            pipeline.create_final_scene_prompts(
                stage1_segments=[
                    {"segment_id": "segment_001"},
                    {"segment_id": "segment_002"},
                ],
                concept_prompts={"segment_001": "concept one"},
                scene_details={"segment_001": {"camera_motion": "static", "character_motion": "singing"}},
                global_context={"subject": "a singer", "story_idea": "", "style": "", "locations": []},
            )

    def test_subject_and_locations_prompt_requests_story_phase_locations(self):
        modules = MusicVideoModulesFake(subject_locations={"subject": "a singer", "actors": [{"id": "singer", "name": "Mara"}]})
        pipeline = MusicVideoPromptPipeline(object(), prompt_modules=modules)

        pipeline.create_subject_and_locations("A quest through caverns, a dragon lair, and a magic spring.")

        self.assertEqual("A quest through caverns, a dragon lair, and a magic spring.", modules.calls[0].payload["story_idea"])

    def test_scene_details_receive_selected_scene_cast(self):
        modules = MusicVideoModulesFake(detail="DETAIL RESULT")
        pipeline = MusicVideoPromptPipeline(object(), prompt_modules=modules)
        progress = []

        pipeline.create_scene_details(
            concept_prompts={
                "segment_001": {
                    "concept": "warrior leads mage through the gate",
                    "references": {"actor_ids": ["warrior", "mage"], "location_id": "gate"},
                }
            },
            stage1_segments=[{"segment_id": "segment_001", "type": "instrumental"}],
            global_context={
                "actors": [{"id": "warrior", "name": "Warrior"}, {"id": "mage", "name": "Mage"}],
                "subject_mode": "multi",
                "max_scene_actors": 4,
            },
            progress_callback=lambda current, total: progress.append((current, total)),
        )

        for call in modules.calls:
            payload = call.payload
            self.assertEqual(["warrior", "mage"], payload["scene_cast"]["visible_actor_ids"])
            self.assertTrue(payload["scene_cast"]["requires_group_staging"])

        self.assertEqual([(1, 1)], progress)

    def test_scene_details_generate_generic_spatial_relations(self):
        modules = MusicVideoModulesFake(detail="SPATIAL DETAIL")
        pipeline = MusicVideoPromptPipeline(object(), prompt_modules=modules)

        details = pipeline.create_scene_details(
            concept_prompts={"segment_001": "A person approaches a doorway."},
            stage1_segments=[{"segment_id": "segment_001", "type": "instrumental"}],
            global_context={},
        )

        self.assertEqual("SPATIAL DETAIL", details["segment_001"]["spatial_relations"])
        spatial_calls = [call for call in modules.calls if call.payload.get("label") in {
            "Camera Motion", "Character Motion", "Spatial Relations"
        }]
        self.assertEqual(3, len(spatial_calls))
        self.assertEqual("Spatial Relations", spatial_calls[-1].payload["label"])


if __name__ == "__main__":
    unittest.main()
