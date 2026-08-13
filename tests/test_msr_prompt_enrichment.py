import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from feverslop.application.msr_prompt_enrichment import enrich_render_plan_with_msr_prompts
from tests.fakellm import (
    FakeLLM,
    FakeVisionLLM,
    FailingVisionLLM,
    FailingVisionAndTextLLM,
    VisionOnlyLLM,
)


class MSRPromptEnrichmentTests(unittest.TestCase):
    def test_rejects_invalid_render_plan_after_write(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            plan = temp / "render_plan.json"
            output = temp / "out.json"
            plan.write_text(json.dumps([{
                "scene": 1,
                "ltx": {"prompt_relay": []},
            }]), encoding="utf-8")

            def write_invalid(path, _data):
                Path(path).write_text("not json", encoding="utf-8")
                return Path(path)

            with patch("feverslop.application.msr_prompt_enrichment.atomic_write_json", side_effect=write_invalid):
                with self.assertRaisesRegex(ValueError, "invalid JSON"):
                    enrich_render_plan_with_msr_prompts(plan, output)

    def test_vision_failure_uses_deterministic_fallback_when_text_completion_is_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            (temp / "mara.png").write_bytes(b"actor")
            plan = temp / "render_plan.json"
            plan.write_text(json.dumps([{
                "scene": 1,
                "references": {
                    "actor_msr_paths": ["mara.png"],
                    "actor_reference_descriptions": [{"id": "mara", "name": "Mara"}],
                },
                "metadata": {"character_motion": "Mara crosses the room."},
                "ltx": {"prompt_relay": [{"frame_start": 0, "frame_end": 9, "state": "instrumental"}]},
            }]))

            for llm in (FailingVisionAndTextLLM("unused"), VisionOnlyLLM()):
                with self.subTest(llm=type(llm).__name__):
                    output = enrich_render_plan_with_msr_prompts(plan, temp / "out.json", llm=llm)
                    prompt = json.loads(output.read_text())[0]["ltx"]["msr_prompt_relay"][0]["prompt"]
                    self.assertIn("Mara stays silent with mouth closed", prompt)
                    self.assertIn("Mara crosses the room", prompt)

    def test_vision_response_supplies_global_descriptions_and_indexed_local_relays(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            (temp / "mara.png").write_bytes(b"actor")
            (temp / "stage.png").write_bytes(b"location")
            plan = temp / "render_plan.json"
            plan.write_text(json.dumps([{
                "scene": 4,
                "references": {
                    "actor_msr_paths": ["mara.png"],
                    "location_msr_path": "stage.png",
                    "actor_reference_descriptions": [{"id": "mara", "name": "Mara", "visual_description": "fallback actor"}],
                    "location_reference_description": {"id": "stage", "name": "Stage", "visual_description": "fallback stage"},
                },
                "ltx": {"prompt_relay": [
                    {"frame_start": 0, "frame_end": 19, "state": "instrumental", "prompt": "old zero"},
                    {"frame_start": 20, "frame_end": 47, "state": "instrumental", "prompt": "old one"},
                ]},
            }]), encoding="utf-8")
            response = json.dumps({
                "references": [
                    {"id": "mara", "type": "actor", "description": "Mara has a sharp black bob, an angular silver jacket, and watchful grey eyes"},
                    {"id": "stage", "type": "location", "description": "A circular rain-dark stage sits beneath cyan rigging and drifting theatrical haze"},
                ],
                "relays": [
                    {"index": 1, "prompt": "Mara crosses into the cyan haze in silence, mouth closed, as the camera arcs behind her and rain trembles across the stage floor."},
                    {"index": 0, "prompt": "Mara waits beneath the rigging in silence, mouth closed, while a slow push-in catches her jacket moving against the wind and haze."},
                ],
            })
            llm = FakeVisionLLM(response)
            statuses = []

            output = enrich_render_plan_with_msr_prompts(plan, temp / "out.json", llm=llm, on_analysis_status=lambda scene, refs: statuses.append((scene, refs)))

            ltx = json.loads(output.read_text(encoding="utf-8"))[0]["ltx"]
            self.assertIn("sharp black bob", ltx["msr_global_prompt"])
            self.assertIn("rain-dark stage", ltx["msr_global_prompt"])
            self.assertEqual([(0, 19), (20, 47)], [(r["frame_start"], r["frame_end"]) for r in ltx["msr_prompt_relay"]])
            self.assertIn("waits beneath", ltx["msr_prompt_relay"][0]["prompt"])
            self.assertIn("crosses into", ltx["msr_prompt_relay"][1]["prompt"])
            for relay in ltx["msr_prompt_relay"]:
                for forbidden in ("Reference Sheet Description", "Target Description", "left panel", "sharp black bob"):
                    self.assertNotIn(forbidden, relay["prompt"])
            self.assertEqual([temp / "mara.png", temp / "stage.png"], llm.calls[0].image_paths)
            self.assertEqual([(4, [{"id": "mara", "type": "actor"}, {"id": "stage", "type": "location"}])], statuses)

    def test_invalid_vision_response_falls_back_to_text_completion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            (temp / "mara.png").write_bytes(b"actor")
            plan = temp / "render_plan.json"
            plan.write_text(json.dumps([{
                "scene": 1,
                "references": {"actor_msr_paths": ["mara.png"], "actor_reference_descriptions": [{"id": "mara", "name": "Mara"}]},
                "ltx": {"prompt_relay": [{"frame_start": 0, "frame_end": 9, "state": "instrumental"}]},
            }]), encoding="utf-8")
            llm = FakeVisionLLM('{"references": [{"id": "wrong", "type": "actor", "description": "wrong"}], "relays": []}')

            output = enrich_render_plan_with_msr_prompts(plan, temp / "out.json", llm=llm)

            relay = json.loads(output.read_text(encoding="utf-8"))[0]["ltx"]["msr_prompt_relay"][0]["prompt"]
            self.assertIn("mouth closed", relay)
            self.assertEqual(2, len(llm.calls))
            self.assertIsNone(llm.calls[1].image_paths)

    def test_missing_images_do_not_attempt_multimodal_completion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            plan = temp / "render_plan.json"
            plan.write_text(json.dumps([{
                "scene": 1,
                "references": {"actor_msr_paths": ["missing.png"], "actor_reference_descriptions": [{"id": "mara", "name": "Mara"}]},
                "ltx": {"prompt_relay": [{"frame_start": 0, "frame_end": 9, "state": "instrumental"}]},
            }]), encoding="utf-8")
            llm = FakeVisionLLM("not json")

            enrich_render_plan_with_msr_prompts(plan, temp / "out.json", llm=llm)

            self.assertEqual(1, len(llm.calls))
            self.assertIsNone(llm.calls[0].image_paths)

    def test_partial_missing_images_keep_full_deterministic_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            (temp / "ivo.png").write_bytes(b"actor")
            (temp / "archive.png").write_bytes(b"location")
            plan = temp / "render_plan.json"
            plan.write_text(json.dumps([{
                "scene": 8,
                "references": {
                    "actor_msr_paths": ["missing-mara.png", "ivo.png"],
                    "location_msr_path": "archive.png",
                    "actor_reference_descriptions": [
                        {"id": "mara", "name": "Mara", "visual_description": "Mara fallback detail"},
                        {"id": "ivo", "name": "Ivo", "visual_description": "Ivo fallback detail"},
                    ],
                    "location_reference_description": {"id": "archive", "name": "Archive", "visual_description": "Archive fallback detail"},
                },
                "ltx": {"prompt_relay": [{"frame_start": 0, "frame_end": 9, "state": "instrumental"}]},
            }]), encoding="utf-8")
            llm = FakeVisionLLM("not json")

            output = enrich_render_plan_with_msr_prompts(plan, temp / "out.json", llm=llm)

            global_prompt = json.loads(output.read_text(encoding="utf-8"))[0]["ltx"]["msr_global_prompt"]
            self.assertIn("Reference image 1: Mara", global_prompt)
            self.assertIn("Reference image 2: Ivo", global_prompt)
            self.assertIn("Reference image 3 (scene): Archive", global_prompt)
            self.assertTrue(all(call.image_paths is None for call in llm.calls))

    def test_transport_exception_is_logged_as_vision_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            (temp / "mara.png").write_bytes(b"actor")
            plan = temp / "render_plan.json"
            plan.write_text(json.dumps([{
                "scene": 3,
                "references": {"actor_msr_paths": ["mara.png"], "actor_reference_descriptions": [{"id": "mara", "name": "Mara"}]},
                "ltx": {"prompt_relay": [{"frame_start": 0, "frame_end": 9, "state": "instrumental"}]},
            }]), encoding="utf-8")

            with self.assertLogs("feverslop.application.msr_prompt_enrichment", level="WARNING") as logs:
                enrich_render_plan_with_msr_prompts(plan, temp / "out.json", llm=FailingVisionLLM("not json"))

            self.assertTrue(any("reason=vision unavailable" in message for message in logs.output))
            self.assertFalse(any("reason=invalid response" in message for message in logs.output))
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
                                    "image_prompt": (
                                        "Full body spectral wolf reference sheet, luminous blue mist fur, "
                                        "glowing white eyes, readable wolf silhouette."
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
                                "image_prompt": (
                                    "Wide environment reference of an ancient stone circle, golden monolith, "
                                    "dense fog and moonlit forest."
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
            self.assertIn("A large translucent wolf made of swirling blue mist", ltx["msr_global_prompt"])
            self.assertNotIn("Full body spectral wolf reference sheet", ltx["msr_global_prompt"])
            self.assertIn("Reference image 2 (scene): Megalith Circle", ltx["msr_global_prompt"])
            self.assertIn("A clearing featuring a massive ancient stone monolith", ltx["msr_global_prompt"])
            self.assertNotIn("Wide environment reference of an ancient stone circle", ltx["msr_global_prompt"])
            self.assertNotIn("Do not duplicate reference subjects", ltx["msr_global_prompt"])
            self.assertEqual(2, len(ltx["msr_prompt_relay"]))
            self.assertIn("Spectral Wolf", ltx["msr_preroll_prompt"])
            self.assertIn("Megalith Circle", ltx["msr_preroll_prompt"])
            self.assertIn("fog", ltx["msr_preroll_prompt"].lower())
            self.assertNotIn("pre-roll", ltx["msr_preroll_prompt"].lower())
            self.assertNotIn("preserve", ltx["msr_preroll_prompt"].lower())
            self.assertIn("Spectral Wolf", ltx["msr_tail_prompt"])
            self.assertIn("Megalith Circle", ltx["msr_tail_prompt"])
            self.assertNotIn("tail safety", ltx["msr_tail_prompt"].lower())
            self.assertIn("Spectral Wolf prowls", ltx["msr_prompt_relay"][0]["prompt"])
            self.assertIn("mouth closed", ltx["msr_prompt_relay"][0]["prompt"])
            self.assertNotIn("lip sync", ltx["msr_prompt_relay"][0]["prompt"].lower())
            self.assertIn("sings the phrase", ltx["msr_prompt_relay"][1]["prompt"])
            self.assertIn("lip sync", ltx["msr_prompt_relay"][1]["prompt"].lower())
            self.assertNotIn("preserve same shot", ltx["msr_prompt_relay"][1]["prompt"].lower())
            self.assertNotIn("Start frame", ltx["msr_prompt_relay"][1]["prompt"])
            self.assertEqual(1, len(llm.calls))
            self.assertIn("Return ONLY valid JSON array", llm.calls[0].system_prompt)

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

    def test_silent_mode_rejects_singing_llm_segments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_plan = temp / "render_plan_refs.json"
            output_plan = temp / "render_plan_refs.json"
            input_plan.write_text(
                json.dumps([
                    {
                        "scene": 2,
                        "metadata": {
                            "type": "vocals",
                            "silent_mode": True,
                            "lyrics": "Do not sing this",
                            "base_concept": "The warrior acts out the story silently.",
                            "camera_motion": "Slow push in.",
                            "character_motion": "The warrior raises one hand and turns away.",
                        },
                        "references": {
                            "actor_reference_descriptions": [
                                {"id": "warrior", "name": "Warrior", "visual_description": "Dark-haired warrior"}
                            ],
                            "location_reference_description": {"id": "void", "name": "Void", "visual_description": "Neon void"},
                        },
                        "ltx": {
                            "prompt_relay": [
                                {
                                    "frame_start": 0,
                                    "frame_end": 48,
                                    "state": "singing",
                                    "prompt": "same scene, character sings with expressive lip sync",
                                }
                            ],
                        },
                    }
                ]),
                encoding="utf-8",
            )
            llm = FakeLLM(json.dumps([{"index": 0, "prompt": "Warrior sings the phrase with clear lip sync."}]))

            result = enrich_render_plan_with_msr_prompts(input_plan, output_plan, llm=llm)

            data = json.loads(result.read_text(encoding="utf-8"))
            relay = data[0]["ltx"]["msr_prompt_relay"][0]
            combined = json.dumps(data[0]["ltx"], ensure_ascii=False).lower()

        self.assertEqual("instrumental", relay["state"])
        self.assertIn("stays silent with mouth closed", relay["prompt"])
        for banned in ("sings", "singing", "lip sync", "lip-sync"):
            self.assertNotIn(banned, combined)

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
