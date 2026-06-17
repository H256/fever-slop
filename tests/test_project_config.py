import json
import tempfile
import unittest
from pathlib import Path

from project_config import ProjectConfig


class ProjectConfigTests(unittest.TestCase):
    def test_lora_1_defaults_to_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            audio = temp / "song.mp3"
            audio.write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "project_name": "test",
                        "input_audio": "song.mp3",
                    }
                ),
                encoding="utf-8",
            )

            config = ProjectConfig.load(config_path)

            self.assertFalse(config.lora_1.enabled)
            self.assertEqual("", config.lora_1.name)
            self.assertEqual(1.0, config.lora_1.strength_model)
            self.assertEqual(1.0, config.lora_1.strength_clip)

    def test_loads_lora_1_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            audio = temp / "song.mp3"
            audio.write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "project_name": "test",
                        "input_audio": "song.mp3",
                        "lora_1": {
                            "enabled": True,
                            "name": "characters/test.safetensors",
                            "strength_model": 0.85,
                            "strength_clip": 0.65,
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = ProjectConfig.load(config_path)

            self.assertTrue(config.lora_1.enabled)
            self.assertEqual("characters/test.safetensors", config.lora_1.name)
            self.assertEqual(0.85, config.lora_1.strength_model)
            self.assertEqual(0.65, config.lora_1.strength_clip)

    def test_loads_zimage_and_ltx_steering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            audio = temp / "song.mp3"
            audio.write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "project_name": "test",
                        "input_audio": "song.mp3",
                        "steering": {
                            "zimage": "z-image steering",
                            "ltx": "ltx steering",
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = ProjectConfig.load(config_path)

            self.assertEqual("z-image steering", config.steering.zimage)
            self.assertEqual("ltx steering", config.steering.ltx)

    def test_loads_prompt_guidance_categories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            audio = temp / "song.mp3"
            audio.write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "project_name": "test",
                        "input_audio": "song.mp3",
                        "prompt_guidance": {
                            "shot_types": "close-up, medium shot",
                            "camera_motion": "slow push-in, handheld orbit",
                            "lighting": "soft rim light",
                            "facial_expression": "focused eyes",
                            "physical_interaction": "raises one hand",
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = ProjectConfig.load(config_path)

            self.assertEqual("close-up, medium shot", config.prompt_guidance.shot_types)
            self.assertEqual("slow push-in, handheld orbit", config.prompt_guidance.camera_motion)
            self.assertEqual("soft rim light", config.prompt_guidance.lighting)
            self.assertEqual("focused eyes", config.prompt_guidance.facial_expression)
            self.assertEqual("raises one hand", config.prompt_guidance.physical_interaction)

    def test_loads_utf8_bom_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            audio = temp / "song.mp3"
            audio.write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "project_name": "test",
                        "input_audio": "song.mp3",
                    }
                ),
                encoding="utf-8-sig",
            )

            config = ProjectConfig.load(config_path)

            self.assertEqual("test", config.project_name)


if __name__ == "__main__":
    unittest.main()
