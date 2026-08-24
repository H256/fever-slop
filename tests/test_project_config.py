import json
import tempfile
import unittest
from pathlib import Path

from feverslop.config.project_config import ProjectConfig


class ProjectConfigTests(unittest.TestCase):
    def test_workflow_config_defaults_to_unselected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            (temp / "song.mp3").write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(json.dumps({"input_audio": "song.mp3"}), encoding="utf-8")

            config = ProjectConfig.load(config_path)

        self.assertIsNone(config.workflows.video)
        self.assertIsNone(config.workflows.reference_hero)
        self.assertIsNone(config.workflows.reference_edit)

    def test_loads_project_workflow_selections(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            (temp / "song.mp3").write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(json.dumps({
                "input_audio": "song.mp3",
                "workflows": {
                    "video": "workflows/video.json",
                    "reference_hero": "workflows/hero.json",
                    "reference_edit": "workflows/edit.json",
                },
            }), encoding="utf-8")

            config = ProjectConfig.load(config_path)

        self.assertEqual("workflows/video.json", config.workflows.video)
        self.assertEqual("workflows/hero.json", config.workflows.reference_hero)
        self.assertEqual("workflows/edit.json", config.workflows.reference_edit)

    def test_project_workflow_selection_must_be_non_empty_string(self):
        for value in ("", 42, False):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temp_dir:
                temp = Path(temp_dir)
                (temp / "song.mp3").write_bytes(b"")
                config_path = temp / "config.json"
                config_path.write_text(json.dumps({
                    "input_audio": "song.mp3",
                    "workflows": {"video": value},
                }), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "workflows.video must be a non-empty string"):
                    ProjectConfig.load(config_path)

    def test_reference_profile_is_not_part_of_project_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            (temp / "song.mp3").write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(json.dumps({
                "input_audio": "song.mp3",
                "reference_profile": "live_concert",
            }), encoding="utf-8")

            config = ProjectConfig.load(config_path)

        self.assertFalse(hasattr(config, "reference_profile"))

    def test_upscale_config_defaults_to_conservative_seedvr2_3b_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            (temp / "song.mp3").write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(json.dumps({"input_audio": "song.mp3"}), encoding="utf-8")

            config = ProjectConfig.load(config_path)

        self.assertFalse(config.upscale.enabled)
        self.assertEqual("seedvr2_3b_int8_convrot.safetensors", config.upscale.model)
        self.assertEqual(0.35, config.upscale.denoise)
        self.assertEqual(4, config.upscale.temporal_overlap)

    def test_upscale_config_accepts_one_target_dimension(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            (temp / "song.mp3").write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(json.dumps({
                "input_audio": "song.mp3",
                "upscale": {"enabled": True, "target_width": 3840},
            }), encoding="utf-8")

            config = ProjectConfig.load(config_path)

        self.assertEqual(3840, config.upscale.target_width)
        self.assertIsNone(config.upscale.target_height)

    def test_upscale_config_accepts_two_target_dimensions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            (temp / "song.mp3").write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(json.dumps({
                "input_audio": "song.mp3",
                "upscale": {"target_width": 3840, "target_height": 2160},
            }), encoding="utf-8")

            config = ProjectConfig.load(config_path)

        self.assertEqual((3840, 2160), (config.upscale.target_width, config.upscale.target_height))
    def test_reference_images_resolution_defaults_to_video_resolution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            (temp / "song.mp3").write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps({
                    "project_name": "test",
                    "input_audio": "song.mp3",
                    "video": {"width": 1920, "height": 1080},
                }),
                encoding="utf-8",
            )

            config = ProjectConfig.load(config_path)

        self.assertEqual((1920, 1080), config.reference_images.resolve(config.video))

    def test_reference_images_resolution_can_override_video_resolution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            (temp / "song.mp3").write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps({
                    "project_name": "test",
                    "input_audio": "song.mp3",
                    "video": {"width": 1920, "height": 1080},
                    "reference_images": {"width": 2048, "height": 1152},
                }),
                encoding="utf-8",
            )

            config = ProjectConfig.load(config_path)

        self.assertEqual((2048, 1152), config.reference_images.resolve(config.video))

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
                    },
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
                    },
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
                    },
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
                    },
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
                    },
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
                    },
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
                    },
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
                    },
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
                            },
                        ],
                        "locations": [
                            {
                                "id": "stage",
                                "name": "Mirror Stage",
                                "visual_description": "black stage with mirrored floor",
                                "image_prompt": "wide shot of mirror stage",
                            },
                        ],
                    },
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
                    },
                ),
                encoding="utf-8",
            )

            config = ProjectConfig.load(config_path)

            self.assertEqual("single", config.subject_mode)
            self.assertEqual(1, config.max_scene_actors)

    def test_minimax_h3_allows_eight_scene_actors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            (temp / "song.mp3").write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(json.dumps({
                "input_audio": "song.mp3",
                "video_pipeline": "minimax-h3-r2v",
                "max_scene_actors": 8,
            }), encoding="utf-8")

            config = ProjectConfig.load(config_path)

        self.assertEqual(8, config.max_scene_actors)

    def test_ltx_msr_rejects_more_than_four_scene_actors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            (temp / "song.mp3").write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(json.dumps({
                "input_audio": "song.mp3",
                "video_pipeline": "ltx_msr",
                "max_scene_actors": 5,
            }), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "between 1 and 4"):
                ProjectConfig.load(config_path)

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
                            },
                        ],
                    },
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
                    },
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
                            },
                        ],
                    },
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
                    },
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
                    },
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
                    },
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
                    },
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
                    },
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
                    },
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

    def test_rejects_non_dict_subsections(self):
        subsection_keys = (
            "video",
            "reference_images",
            "audio",
            "scene_generation",
            "vocal_detection",
            "steering",
            "prompt_guidance",
            "lora_1",
            "minimax_h3_audio_refs",
        )
        for key in subsection_keys:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp_dir:
                temp = Path(temp_dir)
                audio = temp / "song.mp3"
                audio.write_bytes(b"")
                config_path = temp / "config.json"
                config_path.write_text(
                    json.dumps({key: 123, "input_audio": "song.mp3"}),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, f"{key} must be an object"):
                    ProjectConfig.load(config_path)

    def test_rejects_null_subsection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            audio = temp / "song.mp3"
            audio.write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps({"steering": None, "input_audio": "song.mp3"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "steering must be an object"):
                ProjectConfig.load(config_path)

    def test_rejects_non_object_upscale_message_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            audio = temp / "song.mp3"
            audio.write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps({"upscale": True, "input_audio": "song.mp3"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "upscale must be an object"):
                ProjectConfig.load(config_path)

    def test_rejects_non_object_global_assets_message_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            audio = temp / "song.mp3"
            audio.write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps({"global_assets": [1], "input_audio": "song.mp3"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "global_assets must be an object"):
                ProjectConfig.load(config_path)

    def test_rejects_unknown_global_assets_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            audio = temp / "song.mp3"
            audio.write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps({"global_assets": {"casts": []}, "input_audio": "song.mp3"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                r"global_assets contains unknown key\(s\): casts; valid keys are: cast, locations, props, styles",
            ):
                ProjectConfig.load(config_path)

    def test_rejects_null_global_asset_id_before_stringifying(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            audio = temp / "song.mp3"
            audio.write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps({
                    "global_assets": {"cast": [{"asset_id": None}]},
                    "input_audio": "song.mp3",
                }),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "global_cast asset_id is required"):
                ProjectConfig.load(config_path)

    def test_rejects_non_list_actors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            audio = temp / "song.mp3"
            audio.write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps({"actors": "Mara", "input_audio": "song.mp3"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "actors must be an array"):
                ProjectConfig.load(config_path)

    def test_rejects_non_object_actor_entry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            audio = temp / "song.mp3"
            audio.write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps({"actors": ["Mara"], "input_audio": "song.mp3"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, r"actors\[0\] must be an object"):
                ProjectConfig.load(config_path)

    def test_rejects_non_list_locations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            audio = temp / "song.mp3"
            audio.write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps({"locations": 123, "input_audio": "song.mp3"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "locations must be an array"):
                ProjectConfig.load(config_path)

    def test_missing_input_audio_raises_value_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            config_path = temp / "config.json"
            config_path.write_text(json.dumps({}), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "Project config requires an 'input_audio' string field",
            ):
                ProjectConfig.load(config_path)

    def test_null_input_audio_raises_value_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps({"input_audio": None}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "Project config requires an 'input_audio' string field",
            ):
                ProjectConfig.load(config_path)

    def test_numeric_input_audio_raises_value_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps({"input_audio": 123}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "Project config requires an 'input_audio' string field",
            ):
                ProjectConfig.load(config_path)

    def test_blank_input_audio_keeps_no_audio_sentinel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            audio = temp / "song.mp3"
            audio.write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(
                json.dumps({"input_audio": ""}),
                encoding="utf-8",
            )
            config = ProjectConfig.load(config_path)
            self.assertEqual(temp, config.input_audio)


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
        from feverslop.application.full_auto import slugify_project_name as app_slug
        from feverslop.application.movie import slugify_project_name as movie_slug
        from feverslop.domain.slug_utils import slugify_project_name as domain_slug
        from feverslop.domain.slug_utils import slugify_project_name as studio_slug
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
        import tempfile
        from pathlib import Path
        tmp = tempfile.mktemp(suffix=".json")
        Path(tmp).write_text(json.dumps(self.config))
        return tmp


class AudioRefsStemValidationTests(unittest.TestCase):
    """Test ProjectConfig stem validation for minimax_h3_audio_refs."""

    def _mk_config(self, config_dict):
        import json
        import tempfile
        from pathlib import Path
        tmp = tempfile.mktemp(suffix=".json")
        Path(tmp).write_text(json.dumps(config_dict))
        return tmp

    def test_rejects_invalid_audio_ref_stem(self):
        """Invalid stem values raise ValueError."""
        config_path = self._mk_config({
            "input_audio": "song.wav",
            "minimax_h3_audio_refs": {
                "stems": ["vocals", "piano"],
            },
        })
        with self.assertRaises(ValueError) as ctx:
            ProjectConfig.load(config_path)
        self.assertIn("piano", str(ctx.exception))

    def test_accepts_valid_audio_ref_stems(self):
        """Valid stem values load successfully."""
        config_path = self._mk_config({
            "input_audio": "song.wav",
            "minimax_h3_audio_refs": {
                "stems": ["bass", "drums"],
            },
        })
        config = ProjectConfig.load(config_path)
        self.assertEqual(["bass", "drums"], config.minimax_h3_audio_refs.stems)

    def test_audio_ref_stems_defaults_when_missing(self):
        """Missing minimax_h3_audio_refs defaults to ['vocals', 'full_mix']."""
        config_path = self._mk_config({
            "input_audio": "song.wav",
        })
        config = ProjectConfig.load(config_path)
        self.assertEqual(["vocals", "full_mix"], config.minimax_h3_audio_refs.stems)


class ScenePromptWordCountDefaultsTests(unittest.TestCase):
    def test_word_count_defaults_are_shared_across_builder_loader_and_scaffolds(self):
        from feverslop.adapters.full_auto_scaffold import LocalProjectScaffold
        from feverslop.config.project_config import (
            SCENE_PROMPT_WORD_COUNT_MAX,
            SCENE_PROMPT_WORD_COUNT_MIN,
            ProjectConfig,
            PromptGuidanceConfig,
        )
        from feverslop.domain.full_auto import GeneratedSong, SongSpec
        from feverslop.prompting.scene_prompt_builder import scene_prompt_word_limit
        from feverslop.composition.project_repository import movie_default_config
        from feverslop.composition.project_store import ProjectCreateRequest

        # Pin the behavior-preserving decision explicitly (must not drift to 150).
        self.assertEqual((40, 50), (SCENE_PROMPT_WORD_COUNT_MIN, SCENE_PROMPT_WORD_COUNT_MAX))

        # Builder bare-context fallback == dataclass default.
        self.assertEqual(SCENE_PROMPT_WORD_COUNT_MAX, scene_prompt_word_limit({}))
        self.assertEqual(SCENE_PROMPT_WORD_COUNT_MAX, PromptGuidanceConfig().word_count_max)
        self.assertEqual(SCENE_PROMPT_WORD_COUNT_MIN, PromptGuidanceConfig().word_count_min)

        # Loader default for a config WITHOUT a prompt_guidance section.
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            (temp / "song.mp3").write_bytes(b"")
            config_path = temp / "config.json"
            config_path.write_text(json.dumps({"input_audio": "song.mp3"}), encoding="utf-8")
            config = ProjectConfig.load(config_path)
        self.assertEqual(
            (SCENE_PROMPT_WORD_COUNT_MIN, SCENE_PROMPT_WORD_COUNT_MAX),
            (config.prompt_guidance.word_count_min, config.prompt_guidance.word_count_max),
        )
        self.assertEqual(
            SCENE_PROMPT_WORD_COUNT_MAX,
            scene_prompt_word_limit({"prompt_guidance": config.prompt_guidance.as_prompt_context()}),
        )

        # Studio (movie) scaffold default.
        studio_guidance = movie_default_config(
            ProjectCreateRequest(project_type="movie", name="demo"),
        )["prompt_guidance"]
        self.assertEqual(
            (SCENE_PROMPT_WORD_COUNT_MIN, SCENE_PROMPT_WORD_COUNT_MAX),
            (studio_guidance["word_count_min"], studio_guidance["word_count_max"]),
        )

        # Full-auto scaffold default.
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            audio = temp / "source.mp3"
            audio.write_bytes(b"")
            LocalProjectScaffold().create_project(
                projects_dir=temp,
                project_slug="demo",
                spec=SongSpec(
                    title="Demo",
                    tags="pop",
                    lyrics="la la la",
                    bpm=120,
                    duration_seconds=120.0,
                    language="en",
                    keyscale="C major",
                    visual_story_idea="a demo story",
                    visual_style="cinematic",
                ),
                generated_song=GeneratedSong(audio_path=audio, manifest={}),
            )
            scaffold_config = json.loads((temp / "demo" / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(
            (SCENE_PROMPT_WORD_COUNT_MIN, SCENE_PROMPT_WORD_COUNT_MAX),
            (
                scaffold_config["prompt_guidance"]["word_count_min"],
                scaffold_config["prompt_guidance"]["word_count_max"],
            ),
        )


class ScenePromptWordCountValidationTests(unittest.TestCase):
    """Test ProjectConfig prompt_guidance word count bounds validation."""

    def _mk_config(self, config_dict):
        import json
        import tempfile
        from pathlib import Path
        tmp = tempfile.mktemp(suffix=".json")
        Path(tmp).write_text(json.dumps(config_dict))
        return tmp

    def test_rejects_negative_word_count_min(self):
        config_path = self._mk_config({
            "input_audio": "song.wav",
            "prompt_guidance": {"word_count_min": -5},
        })
        with self.assertRaisesRegex(ValueError, "prompt_guidance.word_count_min must be >= 1, got -5"):
            ProjectConfig.load(config_path)

    def test_rejects_negative_word_count_max(self):
        config_path = self._mk_config({
            "input_audio": "song.wav",
            "prompt_guidance": {"word_count_max": -5},
        })
        with self.assertRaisesRegex(ValueError, "prompt_guidance.word_count_max must be >= 1, got -5"):
            ProjectConfig.load(config_path)

    def test_rejects_zero_word_count_min(self):
        config_path = self._mk_config({
            "input_audio": "song.wav",
            "prompt_guidance": {"word_count_min": 0},
        })
        with self.assertRaisesRegex(ValueError, "prompt_guidance.word_count_min must be >= 1, got 0"):
            ProjectConfig.load(config_path)

    def test_rejects_zero_word_count_max(self):
        config_path = self._mk_config({
            "input_audio": "song.wav",
            "prompt_guidance": {"word_count_max": 0},
        })
        with self.assertRaisesRegex(ValueError, "prompt_guidance.word_count_max must be >= 1, got 0"):
            ProjectConfig.load(config_path)

    def test_rejects_crossed_word_count_bounds(self):
        config_path = self._mk_config({
            "input_audio": "song.wav",
            "prompt_guidance": {"word_count_min": 60, "word_count_max": 40},
        })
        with self.assertRaisesRegex(
            ValueError,
            r"prompt_guidance.word_count_min \(60\) must be <= prompt_guidance.word_count_max \(40\)",
        ):
            ProjectConfig.load(config_path)

    def test_accepts_equal_word_count_bounds_below_defaults(self):
        config_path = self._mk_config({
            "input_audio": "song.wav",
            "prompt_guidance": {"word_count_min": 30, "word_count_max": 30},
        })
        config = ProjectConfig.load(config_path)
        self.assertEqual(
            (30, 30),
            (config.prompt_guidance.word_count_min, config.prompt_guidance.word_count_max),
        )


if __name__ == "__main__":
    unittest.main()
