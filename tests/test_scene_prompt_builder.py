import json
import tempfile
import unittest
from pathlib import Path

from feverslop.adapters.local_artifacts import JsonArtifactStore
from feverslop.prompting.scene_prompt_builder import ScenePromptBuilder


class FakeLLM:
    def __init__(self):
        self.calls = []

    def complete_prompt(self, system_prompt: str, prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "prompt": prompt})
        lower = system_prompt.lower()
        if "text-to-image prompt" in lower:
            return "T2I RESULT"
        if "image-to-video prompt" in lower:
            return "I2V RESULT"
        return "DETAIL RESULT"


class ScenePromptBuilderTests(unittest.TestCase):
    def test_scene_prompts_include_explicit_t2i_and_i2v_fields(self):
        llm = FakeLLM()
        builder = ScenePromptBuilder(llm)
        stage1_segments = [
            {
                "segment_id": "segment_001",
                "type": "vocals",
                "start": 0.0,
                "end": 4.0,
                "duration": 4.0,
                "lyrics": "hello",
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "scene_prompts.json"
            builder.build_scene_prompts(
                stage1_segments=stage1_segments,
                concept_prompts={"segment_001": "A lone singer on a mountain peak."},
                scene_details={"segment_001": {"camera_motion": "slow push-in", "character_motion": "hair in the wind"}},
                global_context={
                    "subject": "a man with long hair",
                    "story_idea": "A mountain performance.",
                    "style": "cinematic realism",
                    "locations": ["mountain peak"],
                    "prompt_guidance": {},
                },
                output_json_path=output_path,
                artifact_store=JsonArtifactStore(),
            )

            data = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual("T2I RESULT", data[0]["t2i_prompt"])
        self.assertEqual("T2I RESULT", data[0]["zimage_prompt"])
        self.assertEqual("T2I RESULT", data[0]["ltx_base_prompt"])
        self.assertEqual("I2V RESULT", data[0]["i2v_prompt_from_t2i"])
        self.assertEqual("I2V RESULT", data[0]["original_style_i2v_prompt"])
        self.assertGreaterEqual(len(llm.calls), 2)
        self.assertIn("T2I RESULT", llm.calls[-1]["prompt"])

    def test_zimage_prompt_payload_includes_location_constraint(self):
        llm = FakeLLM()
        builder = ScenePromptBuilder(llm)

        builder.build_zimage_prompt(
            segment={"segment_id": "segment_001", "type": "instrumental"},
            concept="A youth kneels at a spring.",
            global_context={
                "subject": "a Greek youth",
                "story_idea": "A forest myth.",
                "style": "chiaroscuro",
                "locations": ["ancient forest", "secluded spring"],
                "location_constraint": "Allowed locations: ancient forest, secluded spring",
                "prompt_guidance": {},
            },
        )

        payload = json.loads(llm.calls[0]["prompt"])

        self.assertEqual("Allowed locations: ancient forest, secluded spring", payload["location_constraint"])

    def test_build_scene_prompts_carries_reference_metadata_from_concept_dict(self):
        llm = FakeLLM()
        builder = ScenePromptBuilder(llm)
        stage1_segments = [
            {
                "segment_id": "segment_001",
                "type": "vocals",
                "start": 0.0,
                "end": 4.0,
                "duration": 4.0,
                "lyrics": "hello",
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "scene_prompts.json"
            builder.build_scene_prompts(
                stage1_segments=stage1_segments,
                concept_prompts={
                    "segment_001": {
                        "concept": "Mara sings on the mirror stage.",
                        "references": {"actor_ids": ["singer"], "location_id": "stage"},
                    }
                },
                scene_details={},
                global_context={
                    "subject": "Mara",
                    "story_idea": "A stage performance.",
                    "style": "cinematic realism",
                    "locations": ["Mirror Stage"],
                    "location_constraint": "Allowed locations: Mirror Stage",
                    "prompt_guidance": {},
                },
                output_json_path=output_path,
                artifact_store=JsonArtifactStore(),
            )

            data = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual({"actor_ids": ["singer"], "location_id": "stage"}, data[0]["references"])


if __name__ == "__main__":
    unittest.main()
