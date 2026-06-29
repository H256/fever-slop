import json
import tempfile
import unittest
from pathlib import Path

from feverslop.application.msr_prompt_enrichment import enrich_render_plan_with_msr_prompts


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def complete_prompt(self, system_prompt: str, prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "prompt": prompt})
        return self.response


class MSRPromptEnrichmentTests(unittest.TestCase):
    def test_enriches_global_prompt_and_segment_directions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_plan = temp / "render_plan_refs.json"
            output_plan = temp / "render_plan_refs.json"
            input_plan.write_text(
                json.dumps([
                    {
                        "scene": 6,
                        "fps": 24,
                        "frame_count": 85,
                        "metadata": {
                            "type": "mixed",
                            "lyrics": "Durch Nebel und Dom.",
                            "base_concept": (
                                "Thick fog rolls through the Megalith Circle while the Spectral Wolf's "
                                "blue light flickers through the mist."
                            ),
                            "camera_motion": "Slowly panning across the fog-covered stones.",
                            "character_motion": (
                                "The Spectral Wolf prowls slowly through the mist, its body tensing "
                                "and relaxing with each heavy step."
                            ),
                        },
                        "references": {
                            "actor_reference_descriptions": [
                                {
                                    "id": "spectral_wolf",
                                    "name": "Spectral Wolf",
                                    "role": "supernatural antagonist",
                                    "visual_description": (
                                        "A large translucent wolf made of swirling blue mist and white light, "
                                        "with glowing eyes and ethereal fur."
                                    ),
                                }
                            ],
                            "location_reference_description": {
                                "id": "megalith_circle",
                                "name": "Megalith Circle",
                                "visual_description": (
                                    "A clearing featuring a massive ancient stone monolith pulsing with "
                                    "internal golden light."
                                ),
                            },
                        },
                        "ltx": {
                            "base_prompt": "old global",
                            "prompt_relay": [
                                {
                                    "frame_start": 0,
                                    "frame_end": 30,
                                    "state": "instrumental",
                                    "prompt": "same scene, instrumental section, character is not singing",
                                },
                                {
                                    "frame_start": 30,
                                    "frame_end": 84,
                                    "state": "singing",
                                    "prompt": "same scene, character sings with expressive lip sync",
                                },
                            ],
                        },
                    }
                ]),
                encoding="utf-8",
            )
            llm = FakeLLM(json.dumps([
                {
                    "index": 0,
                    "prompt": (
                        "Spectral Wolf prowls through the Megalith Circle fog in silence, mouth closed, "
                        "while the camera slowly pans across the fog-covered stones."
                    ),
                },
                {
                    "index": 1,
                    "prompt": (
                        "Spectral Wolf raises its glowing head and sings the phrase \"Durch Nebel und Dom\" "
                        "with clear lip sync as its blue mist body tenses with each step."
                    ),
                },
            ]))

            result = enrich_render_plan_with_msr_prompts(input_plan, output_plan, llm=llm)

            data = json.loads(result.read_text(encoding="utf-8"))
            ltx = data[0]["ltx"]
            self.assertIn("Reference image 1: Spectral Wolf, supernatural antagonist", ltx["msr_global_prompt"])
            self.assertIn("Reference image 2 (scene): Megalith Circle", ltx["msr_global_prompt"])
            self.assertEqual(2, len(ltx["msr_prompt_relay"]))
            self.assertIn("Spectral Wolf prowls", ltx["msr_prompt_relay"][0]["prompt"])
            self.assertIn("mouth closed", ltx["msr_prompt_relay"][0]["prompt"])
            self.assertNotIn("lip sync", ltx["msr_prompt_relay"][0]["prompt"].lower())
            self.assertIn("sings the phrase", ltx["msr_prompt_relay"][1]["prompt"])
            self.assertIn("lip sync", ltx["msr_prompt_relay"][1]["prompt"].lower())
            self.assertNotIn("preserve same shot", ltx["msr_prompt_relay"][1]["prompt"].lower())
            self.assertNotIn("Start frame", ltx["msr_prompt_relay"][1]["prompt"])
            self.assertEqual(1, len(llm.calls))
            self.assertIn("Return ONLY valid JSON array", llm.calls[0]["system_prompt"])

    def test_falls_back_to_deterministic_direction_when_llm_output_is_invalid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_plan = temp / "render_plan_refs.json"
            output_plan = temp / "render_plan_refs.json"
            input_plan.write_text(
                json.dumps([
                    {
                        "scene": 1,
                        "metadata": {
                            "type": "instrumental",
                            "base_concept": "The wolf circles the stones.",
                            "camera_motion": "Low tracking shot.",
                            "character_motion": "The wolf lowers its head and circles slowly.",
                        },
                        "references": {
                            "actor_reference_descriptions": [
                                {"id": "wolf", "name": "Bactus", "visual_description": "Blue spectral wolf"}
                            ],
                            "location_reference_description": {
                                "id": "circle",
                                "name": "Stone Circle",
                                "visual_description": "Ancient stones in fog",
                            },
                        },
                        "ltx": {
                            "base_prompt": "old global",
                            "prompt_relay": [
                                {
                                    "frame_start": 0,
                                    "frame_end": 24,
                                    "state": "instrumental",
                                    "prompt": "same scene",
                                }
                            ],
                        },
                    }
                ]),
                encoding="utf-8",
            )

            result = enrich_render_plan_with_msr_prompts(input_plan, output_plan, llm=FakeLLM("not json"))

            relay = json.loads(result.read_text(encoding="utf-8"))[0]["ltx"]["msr_prompt_relay"][0]
            self.assertIn("Bactus", relay["prompt"])
            self.assertIn("Low tracking shot", relay["prompt"])
            self.assertIn("The wolf lowers its head", relay["prompt"])
            self.assertNotIn("lip sync", relay["prompt"].lower())

    def test_reports_progress_per_scene(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_plan = temp / "render_plan_refs.json"
            output_plan = temp / "render_plan_refs.json"
            input_plan.write_text(
                json.dumps([
                    {
                        "scene": 1,
                        "references": {"actor_reference_descriptions": [{"name": "Mara"}]},
                        "ltx": {"prompt_relay": []},
                    },
                    {
                        "scene": 2,
                        "references": {"actor_reference_descriptions": [{"name": "Mara"}]},
                        "ltx": {"prompt_relay": []},
                    },
                ]),
                encoding="utf-8",
            )
            events = []

            enrich_render_plan_with_msr_prompts(
                input_plan,
                output_plan,
                on_scene_complete=lambda scene, completed, total: events.append((scene, completed, total)),
            )

            self.assertEqual([(1, 1, 2), (2, 2, 2)], events)


if __name__ == "__main__":
    unittest.main()
