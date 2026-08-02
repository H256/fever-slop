import importlib
import unittest
from pathlib import Path


class PackageLayoutTests(unittest.TestCase):
    def test_feverslop_package_imports_application_boundaries(self):
        modules = [
            "feverslop.application.generate_render_plan",
            "feverslop.adapters.comfyui_rendering",
            "feverslop.domain.render_plan",
            "feverslop.ports.rendering",
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

    def test_feverslop_package_modules_are_real_files_under_src(self):
        import feverslop.application.generate_render_plan as module

        module_path = Path(module.__file__).as_posix()

        self.assertIn("src", module_path)
        self.assertIn("feverslop/application/generate_render_plan.py", module_path)

    def test_config_modules_resolve_under_src_package(self):
        import feverslop.config.project_config as project_config
        import feverslop.config.video_settings as video_settings

        self.assertIn(
            "src/feverslop/config/project_config.py",
            Path(project_config.__file__).as_posix(),
        )
        self.assertIn(
            "src/feverslop/config/video_settings.py",
            Path(video_settings.__file__).as_posix(),
        )

    def test_pipeline_and_prompting_modules_resolve_under_src_package(self):
        import feverslop.pipeline.render_plan_builder as render_plan_builder
        import feverslop.prompting.scene_prompt_builder as scene_prompt_builder

        self.assertIn(
            "src/feverslop/pipeline/render_plan_builder.py",
            Path(render_plan_builder.__file__).as_posix(),
        )
        self.assertIn(
            "src/feverslop/prompting/scene_prompt_builder.py",
            Path(scene_prompt_builder.__file__).as_posix(),
        )

    def test_adapter_modules_resolve_under_src_package(self):
        import feverslop.adapters.comfyui_client as comfyui_client
        import feverslop.adapters.workflow_patcher as workflow_patcher

        self.assertIn(
            "src/feverslop/adapters/comfyui_client.py",
            Path(comfyui_client.__file__).as_posix(),
        )
        self.assertIn(
            "src/feverslop/adapters/workflow_patcher.py",
            Path(workflow_patcher.__file__).as_posix(),
        )


if __name__ == "__main__":
    unittest.main()
