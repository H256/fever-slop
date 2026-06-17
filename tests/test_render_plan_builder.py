import json
import tempfile
import unittest
from pathlib import Path

from render_plan_builder import (
    DetailListPicker,
    build_original_style_i2v_prompt,
    build_render_plan,
)
from video_settings import VideoSettings


class DetailListPickerTests(unittest.TestCase):
    def test_random_pick_is_deterministic_for_scene_and_seed(self):
        picker = DetailListPicker(seed=123)
        items = ["one", "two", "three"]

        self.assertEqual(
            picker.pick("camera_motion", items, scene_number=4, strategy="random"),
            picker.pick("camera_motion", items, scene_number=4, strategy="random"),
        )

    def test_random_no_repeat_cycles_before_reusing_items(self):
        picker = DetailListPicker(seed=99)
        items = ["one", "two", "three"]

        first_cycle = [
            picker.pick("camera_motion", items, scene_number=i, strategy="random_no_repeat")
            for i in range(1, 4)
        ]
        fourth = picker.pick("camera_motion", items, scene_number=4, strategy="random_no_repeat")

        self.assertEqual(set(items), set(first_cycle))
        self.assertIn(fourth, items)


class OriginalStylePromptTests(unittest.TestCase):
    def test_vocal_prompt_contains_singing_but_no_silent_motion(self):
        prompt = build_original_style_i2v_prompt(
            scene={
                "scene": 1,
                "type": "vocals",
                "ltx_base_prompt": "A singer in a red jacket stands under neon rain.",
                "base_concept": "neon rain performance",
            },
            seed=7,
        ).lower()

        self.assertIn("sings", prompt)
        self.assertIn("lip sync", prompt)
        self.assertNotIn("no lip", prompt)
        self.assertLess(prompt.index("singer"), 80)

    def test_instrumental_prompt_excludes_singing_terms(self):
        prompt = build_original_style_i2v_prompt(
            scene={
                "scene": 2,
                "type": "instrumental",
                "ltx_base_prompt": "A guitarist waits beside a wall of amber lights.",
                "base_concept": "quiet instrumental break",
            },
            seed=7,
        ).lower()

        for banned in ("sing", "sings", "singing", "lip sync", "lip-sync"):
            self.assertNotIn(banned, prompt)


class BuildRenderPlanTests(unittest.TestCase):
    def test_render_plan_includes_original_style_prompt_and_mode_hints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scene_prompts_path = temp / "scene_prompts.json"
            relay_path = temp / "relay.json"
            output_path = temp / "render_plan.json"

            scene_prompts_path.write_text(
                json.dumps(
                    [
                        {
                            "scene": 1,
                            "segment_id": "s1",
                            "type": "vocals",
                            "start": 0.0,
                            "end": 2.0,
                            "duration": 2.0,
                            "lyrics": "hello",
                            "base_concept": "stage",
                            "zimage_prompt": "z1",
                            "ltx_base_prompt": "A singer faces the camera on a small stage.",
                        },
                        {
                            "scene": 16,
                            "segment_id": "s16",
                            "type": "vocals",
                            "start": 2.0,
                            "end": 4.0,
                            "duration": 2.0,
                            "lyrics": "mixed line",
                            "base_concept": "stage",
                            "zimage_prompt": "z16",
                            "ltx_base_prompt": "The same singer remains framed on the stage.",
                        },
                        {
                            "scene": 3,
                            "segment_id": "s3",
                            "type": "instrumental",
                            "start": 4.0,
                            "end": 6.0,
                            "duration": 2.0,
                            "lyrics": "",
                            "base_concept": "instrumental",
                            "zimage_prompt": "z3",
                            "ltx_base_prompt": "The performer holds still in a spotlight.",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            relay_path.write_text(
                json.dumps(
                    [
                        {"scene": 1, "prompt_relay": []},
                        {
                            "scene": 16,
                            "prompt_relay": [
                                {"frame_start": 0, "frame_end": 20, "state": "singing", "lyrics": "mixed"},
                                {"frame_start": 20, "frame_end": 30, "state": "instrumental"},
                                {"frame_start": 30, "frame_end": 48, "state": "singing", "lyrics": "line"},
                            ],
                        },
                        {"scene": 3, "prompt_relay": []},
                    ]
                ),
                encoding="utf-8",
            )

            build_render_plan(
                scene_prompts_json=scene_prompts_path,
                ltx_prompt_relay_json=relay_path,
                output_json_file=output_path,
                video_settings=VideoSettings(fps=24, width=1280, height=704),
            )

            plan = json.loads(output_path.read_text(encoding="utf-8"))
            by_scene = {item["scene"]: item for item in plan}

            self.assertTrue(all(item["ltx"].get("original_style_i2v_prompt") for item in plan))
            self.assertEqual("single_prompt", by_scene[1]["ltx"]["render_mode_hint"])
            self.assertEqual("relay", by_scene[16]["ltx"]["render_mode_hint"])
            self.assertEqual("single_prompt", by_scene[3]["ltx"]["render_mode_hint"])
            self.assertNotIn("lip sync", by_scene[3]["ltx"]["original_style_i2v_prompt"].lower())


if __name__ == "__main__":
    unittest.main()
