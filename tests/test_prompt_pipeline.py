import json
import unittest

from feverslop.prompting.prompt_pipeline import MusicVideoPromptPipeline


class FakeConceptLLM:
    def __init__(self):
        self.calls = []

    def complete_prompt(self, system_prompt: str, prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "prompt": prompt})
        return '{"segment_001": "A youth stands in the allowed forest."}'


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


if __name__ == "__main__":
    unittest.main()
