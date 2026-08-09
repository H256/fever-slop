import json
import unittest

from feverslop.prompting.prompt_pipeline import MusicVideoPromptPipeline


class FakeConceptLLM:
    def __init__(self):
        self.calls = []

    def complete_prompt(self, system_prompt: str, prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "prompt": prompt})
        return '{"segment_001": "A youth stands in the allowed forest."}'


class FakeSubjectLocationLLM:
    def __init__(self):
        self.calls = []

    def complete_prompt(self, system_prompt: str, prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "prompt": prompt})
        return '{"subject": "a singer", "actors": [{"id": "singer", "name": "Mara"}], "locations": ["stage"]}'


class MusicVideoPromptPipelineTests(unittest.TestCase):
    def test_create_concept_prompts_includes_global_context_and_notes(self):
        llm = FakeConceptLLM()
        pipeline = MusicVideoPromptPipeline(llm)

        pipeline.create_concept_prompts(
            stage1_segments=[{"segment_id": "segment_001", "type": "instrumental"}],
            story_idea="A forest myth.",
            global_context={
                "locations": ["ancient forest"],
                "location_constraint": "Allowed locations: ancient forest",
            },
            notes="Keep the spring visible.",
        )

        payload = json.loads(llm.calls[0]["prompt"])

        self.assertEqual("Allowed locations: ancient forest", payload["GLOBAL_CONTEXT"]["location_constraint"])
        self.assertEqual("Keep the spring visible.", payload["NOTES"])
        self.assertIn("references", llm.calls[0]["system_prompt"])

    def test_subject_and_locations_prompt_requests_multi_actor_reference_data(self):
        llm = FakeSubjectLocationLLM()
        pipeline = MusicVideoPromptPipeline(llm)

        result = pipeline.create_subject_and_locations("stage story")

        self.assertIn('"actors"', llm.calls[0]["system_prompt"])
        self.assertEqual("singer", result["actors"][0]["id"])

    def test_location_image_prompt_avoids_and_removes_reference_sheet_wording(self):
        class ReferenceSheetLLM:
            def complete_prompt(self, system_prompt: str, prompt: str) -> str:
                self.system_prompt = system_prompt
                return '{"subject": "a singer", "actors": [], "locations": [{"id": "stage", "image_prompt": "Cinematic environment reference sheet for a dark stage"}]}'

        llm = ReferenceSheetLLM()
        result = MusicVideoPromptPipeline(llm).create_subject_and_locations("stage story")

        self.assertNotIn("location reference sheet", llm.system_prompt.lower())
        self.assertIn("never use \"environment reference sheet\"", llm.system_prompt.lower())
        self.assertNotIn("reference sheet", result["locations"][0]["image_prompt"].lower())

    def test_create_final_scene_prompts_raises_for_missing_segments(self):
        """Missing segment IDs should produce a clear error, not a raw KeyError."""
        llm = FakeConceptLLM()
        pipeline = MusicVideoPromptPipeline(llm)

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
        llm = FakeSubjectLocationLLM()
        pipeline = MusicVideoPromptPipeline(llm)

        pipeline.create_subject_and_locations("A quest through caverns, a dragon lair, and a magic spring.")

        system_prompt = llm.calls[0]["system_prompt"]
        self.assertIn("major story phases", system_prompt)
        self.assertIn("avoid collapsing", system_prompt.lower())

    def test_scene_details_receive_selected_scene_cast(self):
        llm = FakeConceptLLM()
        pipeline = MusicVideoPromptPipeline(llm)
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

        for call in llm.calls:
            payload = json.loads(call["prompt"])
            self.assertEqual(["warrior", "mage"], payload["scene_cast"]["visible_actor_ids"])
            self.assertTrue(payload["scene_cast"]["requires_group_staging"])

        self.assertEqual([(1, 1)], progress)


if __name__ == "__main__":
    unittest.main()
