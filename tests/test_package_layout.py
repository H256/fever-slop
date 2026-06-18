import importlib
import unittest
from pathlib import Path


class PackageLayoutTests(unittest.TestCase):
    def test_autoprompter_package_imports_application_boundaries(self):
        modules = [
            "autoprompter.application.generate_render_plan",
            "autoprompter.adapters.comfyui_rendering",
            "autoprompter.domain.render_plan",
            "autoprompter.ports.rendering",
        ]

        for module_name in modules:
            with self.subTest(module=module_name):
                self.assertIsNotNone(importlib.import_module(module_name))

    def test_root_compatibility_imports_remain_available(self):
        modules = [
            "main",
            "render_ltx",
            "render_storyboard",
            "ltx_video_renderer",
            "storyboard_renderer",
            "workflow_patcher",
        ]

        for module_name in modules:
            with self.subTest(module=module_name):
                self.assertIsNotNone(importlib.import_module(module_name))

    def test_autoprompter_package_modules_are_real_files_under_src(self):
        import autoprompter.application.generate_render_plan as module

        module_path = Path(module.__file__).as_posix()

        self.assertIn("src", module_path)
        self.assertIn("autoprompter/application/generate_render_plan.py", module_path)

    def test_config_modules_resolve_under_src_package(self):
        import autoprompter.config.project_config as project_config
        import autoprompter.config.video_settings as video_settings

        self.assertIn(
            "src/autoprompter/config/project_config.py",
            Path(project_config.__file__).as_posix(),
        )
        self.assertIn(
            "src/autoprompter/config/video_settings.py",
            Path(video_settings.__file__).as_posix(),
        )

    def test_pipeline_and_prompting_modules_resolve_under_src_package(self):
        import autoprompter.pipeline.render_plan_builder as render_plan_builder
        import autoprompter.prompting.scene_prompt_builder as scene_prompt_builder

        self.assertIn(
            "src/autoprompter/pipeline/render_plan_builder.py",
            Path(render_plan_builder.__file__).as_posix(),
        )
        self.assertIn(
            "src/autoprompter/prompting/scene_prompt_builder.py",
            Path(scene_prompt_builder.__file__).as_posix(),
        )

    def test_adapter_and_audio_modules_resolve_under_src_package(self):
        import autoprompter.adapters.comfyui_client as comfyui_client
        import autoprompter.adapters.workflow_patcher as workflow_patcher
        import autoprompter.audio.beat_analysis as beat_analysis

        self.assertIn(
            "src/autoprompter/adapters/comfyui_client.py",
            Path(comfyui_client.__file__).as_posix(),
        )
        self.assertIn(
            "src/autoprompter/adapters/workflow_patcher.py",
            Path(workflow_patcher.__file__).as_posix(),
        )
        self.assertIn(
            "src/autoprompter/audio/beat_analysis.py",
            Path(beat_analysis.__file__).as_posix(),
        )


if __name__ == "__main__":
    unittest.main()
