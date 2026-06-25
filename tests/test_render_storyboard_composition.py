import tempfile
import unittest
from pathlib import Path


class RenderStoryboardCompositionTests(unittest.TestCase):
    def test_matching_storyboard_prompt_transform_accepts_suffix_workflow_path(self):
        from feverslop.composition.render_storyboard import matching_storyboard_prompt_transform
        from feverslop.config.app_config import (
            AppConfig,
            ComfyUIConfig,
            LLMConfig,
            StoryboardPromptTransformConfig,
        )

        config = AppConfig(
            llm=LLMConfig(),
            comfyui=ComfyUIConfig(),
            storyboard_prompt_transforms=[
                StoryboardPromptTransformConfig(
                    workflow="workflows/image_t2i_startframe_ideogram_v1.json",
                    kind="template",
                    template="docs/ideogram4_prompt_template.md",
                    positive_prompt_input="text",
                    debug_dir="ideogram4_prompt_debug",
                )
            ],
        )

        transform = matching_storyboard_prompt_transform(
            config,
            Path(tempfile.gettempdir()) / "repo" / "workflows" / "image_t2i_startframe_ideogram_v1.json",
        )

        self.assertIsNotNone(transform)
        self.assertEqual("text", transform.positive_prompt_input)

    def test_matching_storyboard_prompt_transform_accepts_windows_relative_workflow_path(self):
        from feverslop.composition.render_storyboard import matching_storyboard_prompt_transform
        from feverslop.config.app_config import (
            AppConfig,
            ComfyUIConfig,
            LLMConfig,
            StoryboardPromptTransformConfig,
        )

        config = AppConfig(
            llm=LLMConfig(),
            comfyui=ComfyUIConfig(),
            storyboard_prompt_transforms=[
                StoryboardPromptTransformConfig(
                    workflow="workflows/image_t2i_startframe_ideogram_v1.json",
                    template="docs/ideogram4_prompt_template.md",
                )
            ],
        )

        transform = matching_storyboard_prompt_transform(
            config,
            r".\workflows\image_t2i_startframe_ideogram_v1.json",
        )

        self.assertIsNotNone(transform)


if __name__ == "__main__":
    unittest.main()
