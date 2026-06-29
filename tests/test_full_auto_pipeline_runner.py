import unittest
from pathlib import Path

from feverslop.composition.pipeline_runner import build_arg_parser


class FullAutoPipelineRunnerTests(unittest.TestCase):
    def test_run_pipeline_adapter_forwards_full_runner_override_surface(self):
        from feverslop.adapters.pipeline_runner import RunPipelineAdapter

        captured = {}

        class Result:
            final_video_path = Path("final.mp4")

        def fake_run(args):
            captured.update(vars(args))
            return Result()

        options = {
            "app_config": "custom_app.json",
            "concept_batch_size": 5,
            "storyboard_workflow": "storyboard.json",
            "reference_hero_workflow": "hero.json",
            "reference_edit_workflow": "edit.json",
            "msr_workflow": "msr.json",
            "relay_workflow": "relay.json",
            "single_prompt_workflow": "single.json",
            "render_mode": "auto",
            "single_prompt_title": "#PROMPT_POSITIVE",
            "single_prompt_input": "text",
            "rolling_frame_profile": "safe",
            "storyboard_lora_strength": 0.4,
            "video_character_lora_strength": 0.8,
            "video_lora_1_strength_model": 0.7,
            "video_lora_1_strength_clip": 0.6,
            "lora_split_enabled": True,
            "randomize_seed": True,
            "smoke_scene": 3,
            "smoke_only": True,
            "no_skip_existing": True,
            "skip_tests": True,
            "skip_main_pipeline": True,
            "skip_relay_compact": True,
            "skip_anchor_fix": True,
            "skip_storyboard": True,
            "skip_storyboard_page": True,
            "skip_msr_reference_render": True,
            "skip_ltx": True,
            "skip_final_concat": True,
            "diagnostic_original_audio_mux": True,
            "no_original_audio_mux": True,
        }

        final = RunPipelineAdapter(
            run_pipeline=fake_run,
            build_arg_parser=build_arg_parser,
        ).run(
            project_config_path=Path("project/config.json"),
            options=options,
        )

        self.assertEqual(Path("final.mp4"), final)
        self.assertEqual(str(Path("project/config.json")), captured["project_config"])
        for key, expected in options.items():
            self.assertEqual(expected, captured[key], key)


if __name__ == "__main__":
    unittest.main()
