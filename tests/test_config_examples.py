import json
import tempfile
import unittest
from pathlib import Path

from feverslop.config.app_config import AppConfig
from feverslop.config.project_config import ProjectConfig


ROOT = Path(__file__).resolve().parents[1]


class ConfigExampleTests(unittest.TestCase):
    def test_project_example_covers_supported_keys_and_loads(self):
        raw = json.loads((ROOT / "config.example.json").read_text(encoding="utf-8"))
        self.assertEqual(
            {
                "project_name", "input_audio", "silent_mode", "lyrics", "video_pipeline",
                "render_profile", "reference_generation", "subject_mode", "max_scene_actors",
                "video", "workflows", "upscale", "reference_images", "audio",
                "scene_generation", "vocal_detection", "story_idea", "style", "music_style",
                "subject", "locations", "actors", "global_assets", "global_cast",
                "global_locations", "global_styles", "global_props", "steering",
                "prompt_guidance", "lora_1", "loras", "lora_split_enabled",
                "minimax_h3_audio_refs",
            },
            set(raw),
        )
        self.assertEqual(
            {"video", "reference_hero", "reference_edit", "reference_sequence"},
            set(raw["workflows"]),
        )
        self.assertEqual(
            {
                "enabled", "workflow_path", "model", "vae", "target_width", "target_height",
                "default_scale", "strategy", "max_pass_scale", "max_ai_passes", "denoise",
                "temporal_overlap", "split_latent", "vae_temporal_size", "vae_temporal_overlap",
                "segment_duration_seconds", "color_correction", "seed",
            },
            set(raw["upscale"]),
        )
        self.assertEqual({"cast", "locations", "styles", "props"}, set(raw["global_assets"]))
        self.assertEqual(
            {"enabled", "name", "strength_model", "strength_clip"}, set(raw["lora_1"])
        )
        self.assertEqual({"stems"}, set(raw["minimax_h3_audio_refs"]))
        self.assertEqual(
            {"id", "name", "role", "visual_description", "image_prompt"},
            set(raw["actors"][0]),
        )
        self.assertEqual(
            {"id", "name", "visual_description", "image_prompt", "reference_mode"},
            set(raw["locations"][0]),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            config = ProjectConfig.load(path)
        self.assertEqual("ltx25-i2v-draft", config.render_profile)
        self.assertEqual("hero", config.actors[0].id)

    def test_app_example_covers_vram_handoff_and_supported_keys(self):
        raw = json.loads((ROOT / "app_config.example.json").read_text(encoding="utf-8"))
        self.assertEqual(
            {
                "llm", "comfyui", "execution", "global_library_path",
                "video_workflow_profiles", "storyboard_prompt_transforms",
            },
            set(raw),
        )
        self.assertEqual(
            {
                "api_key", "base_url", "model", "models", "temperature", "dspy_temperature",
                "max_tokens", "request_timeout_seconds", "dspy_cache", "max_concurrent_requests",
                "prompt_judge_attempts", "prompt_judge_max_tokens",
            },
            set(raw["llm"]),
        )
        self.assertEqual(
            {
                "base_url", "prompt_timeout_seconds", "model_overrides",
                "default_max_render_duration_seconds", "video_workflow_limits",
            },
            set(raw["comfyui"]),
        )
        self.assertEqual({"vram_handoff"}, set(raw["execution"]))
        self.assertEqual(
            {"workflow", "kind", "template", "positive_prompt_input", "debug_dir", "max_words"},
            set(raw["storyboard_prompt_transforms"][0]),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "app_config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            config = AppConfig.load(path)
        self.assertEqual("continuous", config.execution.vram_handoff.value)
        self.assertEqual(150, config.storyboard_prompt_transforms[0].max_words)


if __name__ == "__main__":
    unittest.main()
