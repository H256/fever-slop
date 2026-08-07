import json
import tempfile
import unittest
from pathlib import Path

from feverslop.config.project_config import ProjectConfig


class ProjectConfigTests(unittest.TestCase):
    def test_silent_mode_defaults_to_false(self):
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

            self.assertFalse(config.silent_mode)

    def test_music_video_language_and_seed_defaults(self):
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

            self.assertEqual("en", config.audio.language)
            self.assertEqual(-1, config.scene_generation.seed)

    def test_loads_silent_mode(self):
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
                        "silent_mode": True,
                    }
                ),
                encoding="utf-8",
            )

            config = ProjectConfig.load(config_path)

            self.assertTrue(config.silent_mode)

    def test_silent_mode_must_be_boolean(self):
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
                        "silent_mode": "true",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                ProjectConfig.load(config_path)

    def test_null_silent_mode_defaults_to_false(self):
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
                        "silent_mode": None,
                    }
                ),
                encoding="utf-8",
            )

            config = ProjectConfig.load(config_path)

            self.assertFalse(config.silent_mode)

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

    def test_loads_loras_array_and_split_flag(self):
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
                        "lora_split_enabled": True,
                        "loras": [
                            {
                                "enabled": True,
                                "name": "characters/first.safetensors",
                                "strength_model": 0.8,
                                "strength_clip": 0.6,
                            },
                            {
                                "enabled": False,
                                "name": "characters/second.safetensors",
                                "strength_model": 0.4,
                                "strength_clip": 0.3,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            config = ProjectConfig.load(config_path)

            self.assertTrue(config.lora_split_enabled)
            self.assertEqual(2, len(config.loras))
            self.assertTrue(config.loras[0].enabled)
            self.assertEqual("characters/first.safetensors", config.loras[0].name)
            self.assertEqual(0.8, config.loras[0].strength_model)
            self.assertEqual(0.6, config.loras[0].strength_clip)
            self.assertTrue(config.loras[0].name_explicit)
            self.assertTrue(config.loras[0].strength_model_explicit)
            self.assertTrue(config.loras[0].strength_clip_explicit)
            self.assertFalse(config.loras[1].enabled)
            self.assertEqual("characters/second.safetensors", config.loras[1].name)

    def test_loads_actor_and_structured_location_bible_inputs(self):
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
                        "actors": [
                            {
                                "id": "singer",
                                "name": "Mara",
                                "role": "lead singer",
                                "visual_description": "short silver hair",
                                "image_prompt": "portrait of Mara",
                            }
                        ],
                        "locations": [
                            {
                                "id": "stage",
                                "name": "Mirror Stage",
                                "visual_description": "black stage with mirrored floor",
                                "image_prompt": "wide shot of mirror stage",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            config = ProjectConfig.load(config_path)

            self.assertEqual("singer", config.actors[0].id)
            self.assertEqual("Mara", config.actors[0].name)
            self.assertEqual("portrait of Mara", config.actors[0].image_prompt)
            self.assertEqual(["Mirror Stage"], config.locations)
            self.assertEqual("stage", config.structured_locations[0].id)
            self.assertEqual("wide shot of mirror stage", config.structured_locations[0].image_prompt)

    def test_loads_single_subject_reference_mode(self):
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
                        "subject_mode": "single",
                        "max_scene_actors": 1,
                    }
                ),
                encoding="utf-8",
            )

            config = ProjectConfig.load(config_path)

            self.assertEqual("single", config.subject_mode)
            self.assertEqual(1, config.max_scene_actors)

    def test_loras_array_tracks_omitted_optional_patch_fields(self):
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
                        "loras": [
                            {
                                "enabled": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            config = ProjectConfig.load(config_path)

            self.assertEqual(1, len(config.loras))
            self.assertTrue(config.loras[0].enabled)
            self.assertEqual("", config.loras[0].name)
            self.assertEqual(1.0, config.loras[0].strength_model)
            self.assertEqual(1.0, config.loras[0].strength_clip)
            self.assertFalse(config.loras[0].name_explicit)
            self.assertFalse(config.loras[0].strength_model_explicit)
            self.assertFalse(config.loras[0].strength_clip_explicit)

    def test_lora_1_is_fallback_when_loras_array_is_missing(self):
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
                            "name": "characters/legacy.safetensors",
                            "strength_model": 0.75,
                            "strength_clip": 0.5,
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = ProjectConfig.load(config_path)

            self.assertEqual(1, len(config.loras))
            self.assertTrue(config.loras[0].enabled)
            self.assertEqual("characters/legacy.safetensors", config.loras[0].name)
            self.assertEqual(0.75, config.loras[0].strength_model)

    def test_loras_array_wins_over_legacy_lora_1(self):
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
                            "name": "characters/legacy.safetensors",
                            "strength_model": 0.75,
                            "strength_clip": 0.5,
                        },
                        "loras": [
                            {
                                "enabled": True,
                                "name": "characters/new.safetensors",
                                "strength_model": 0.9,
                                "strength_clip": 0.7,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            config = ProjectConfig.load(config_path)

            self.assertEqual(1, len(config.loras))
            self.assertEqual("characters/new.safetensors", config.loras[0].name)
            self.assertEqual(0.9, config.loras[0].strength_model)

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

    def test_windows_relative_input_audio_is_resolved_from_config_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "project_name": "test",
                        "input_audio": "input\\song.mp3",
                    }
                ),
                encoding="utf-8",
            )

            config = ProjectConfig.load(config_path)

        self.assertEqual(temp / "input" / "song.mp3", config.input_audio)


class ProjectConfigLyricsTests(unittest.TestCase):
    def test_loads_lyrics_string(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input_audio": "input/song.wav",
                        "lyrics": "[Verse]\nfirst line\nsecond line",
                    }
                ),
                encoding="utf-8",
            )

            config = ProjectConfig.load(config_path)

            self.assertEqual("[Verse]\nfirst line\nsecond line", config.lyrics)

    def test_loads_lyrics_list_as_newline_joined_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input_audio": "input/song.wav",
                        "lyrics": ["[Verse]", "first line", "second line"],
                    }
                ),
                encoding="utf-8",
            )

            config = ProjectConfig.load(config_path)

            self.assertEqual("[Verse]\nfirst line\nsecond line", config.lyrics)


class ProjectConfigValidationTests(unittest.TestCase):
    def test_rejects_invalid_fps_type(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            audio = temp / "song.mp3"
            audio.write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps({
                    "project_name": "test",
                    "input_audio": "song.mp3",
                    "video": {"fps": "bad"},
                }),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                ProjectConfig.load(config_path)
            self.assertIn("fps", str(ctx.exception).lower())

    def test_rejects_invalid_width_type(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            audio = temp / "song.mp3"
            audio.write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps({
                    "project_name": "test",
                    "input_audio": "song.mp3",
                    "video": {"width": "bad"},
                }),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                ProjectConfig.load(config_path)
            self.assertIn("width", str(ctx.exception).lower())

    def test_rejects_invalid_height_type(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            audio = temp / "song.mp3"
            audio.write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps({
                    "project_name": "test",
                    "input_audio": "song.mp3",
                    "video": {"height": 0},
                }),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                ProjectConfig.load(config_path)
            self.assertIn("height", str(ctx.exception).lower())


class SlugifyProjectNameTests(unittest.TestCase):
    def test_slugify_normalizes_spaces_and_special_chars(self):
        from feverslop.domain.slug_utils import slugify_project_name
        self.assertEqual("my-song-video", slugify_project_name("My Song Video!"))

    def test_slugify_handles_empty_and_whitespace(self):
        from feverslop.domain.slug_utils import slugify_project_name
        self.assertEqual("", slugify_project_name(""))
        self.assertEqual("", slugify_project_name("   "))

    def test_slugify_collapses_multiple_separators(self):
        from feverslop.domain.slug_utils import slugify_project_name
        self.assertEqual("my-project", slugify_project_name("my___project"))
        self.assertEqual("my-project", slugify_project_name("my---project"))

    def test_slugify_imports_are_consistent(self):
        from feverslop.domain.slug_utils import slugify_project_name as domain_slug
        from feverslop.studio.projects import slugify_project_name as studio_slug
        from feverslop.application.full_auto import slugify_project_name as app_slug
        from feverslop.application.movie import slugify_project_name as movie_slug
        test_cases = ["Hello World", "test_project", "My Song!", ""]
        for case in test_cases:
            self.assertEqual(domain_slug(case), studio_slug(case))
            self.assertEqual(domain_slug(case), app_slug(case))
            self.assertEqual(domain_slug(case), movie_slug(case))



class VideoPipelineFieldTests(unittest.TestCase):
    """Test ProjectConfig.video_pipeline field."""

    def test_defaults_to_ltx_i2v(self):
        """Default value is ltx_i2v."""
        self.config = {"input_audio": "song.wav"}
        config = ProjectConfig.load(self._mk_config())
        self.assertEqual(config.video_pipeline, "ltx_i2v")

    def test_loads_r2v_pipeline(self):
        """Loads minimax-h3-r2v from config."""
        self.config = {"input_audio": "song.wav", "video_pipeline": "minimax-h3-r2v"}
        config = ProjectConfig.load(self._mk_config())
        self.assertEqual(config.video_pipeline, "minimax-h3-r2v")

    def test_loads_msr_pipeline(self):
        self.config = {"input_audio": "song.wav", "video_pipeline": "ltx_msr"}
        config = ProjectConfig.load(self._mk_config())
        self.assertEqual(config.video_pipeline, "ltx_msr")

    def test_empty_string_defaults_to_ltx_i2v(self):
        """Empty string falls back to ltx_i2v."""
        self.config = {"input_audio": "song.wav", "video_pipeline": ""}
        config = ProjectConfig.load(self._mk_config())
        self.assertEqual(config.video_pipeline, "ltx_i2v")

    def _mk_config(self):
        import json
        from pathlib import Path
        import tempfile
        tmp = tempfile.mktemp(suffix=".json")
        Path(tmp).write_text(json.dumps(self.config))
        return tmp


if __name__ == "__main__":
    unittest.main()
