import tempfile
import unittest
from pathlib import Path

from tests.prompt_fakes import GeneralModulesFake


class StoryboardPromptTransformerTests(unittest.TestCase):
    def test_template_transformer_trims_overlength_response_to_configured_limit(self):
        from feverslop.prompting.storyboard_prompt_transformer import (
            TemplateStoryboardPromptTransformer,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            template = temp / "template.md"
            template.write_text("[SYSTEM]\nS\n\n[USER]\nU", encoding="utf-8")
            result = TemplateStoryboardPromptTransformer(
                llm=object(),
                modules=GeneralModulesFake(storyboard="one two three four five"),
                template_path=template,
                debug_dir=temp / "debug",
                max_words=3,
            ).transform_prompt(
                scene_number=1,
                original_prompt="idea",
                width=1280,
                height=704,
            )

        self.assertEqual("one two three", result)

    def test_template_transformer_passes_system_and_filled_user_prompt_to_llm(self):
        from feverslop.prompting.storyboard_prompt_transformer import (
            TemplateStoryboardPromptTransformer,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            template = temp / "template.md"
            template.write_text(
                "[META]\nignored\n\n[SYSTEM]\nSystem rules\n\n[USER]\n"
                "TARGET IMAGE ASPECT RATIO: {{width}}:{{height}} (width:height).\n"
                "User idea: {{original_prompt}}\n",
                encoding="utf-8",
            )
            modules = GeneralModulesFake(storyboard=" raw non-json response \n")

            result = TemplateStoryboardPromptTransformer(
                llm=object(),
                modules=modules,
                template_path=template,
                debug_dir=temp / "debug",
            ).transform_prompt(
                scene_number=1,
                original_prompt="cinematic frame",
                width=1920,
                height=1088,
            )

        self.assertEqual("raw non-json response", result)
        self.assertEqual("System rules", modules.calls[0].payload["system_template"])
        self.assertEqual("TARGET IMAGE ASPECT RATIO: 1920:1088 (width:height).\nUser idea: cinematic frame", modules.calls[0].payload["user_template"])

    def test_template_transformer_writes_debug_files(self):
        from feverslop.prompting.storyboard_prompt_transformer import (
            TemplateStoryboardPromptTransformer,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            template = temp / "template.md"
            template.write_text("[SYSTEM]\nS\n\n[USER]\nU {{width}} {{height}} {{original_prompt}}", encoding="utf-8")

            TemplateStoryboardPromptTransformer(
                llm=object(),
                modules=GeneralModulesFake(storyboard="{not json but ok}"),
                template_path=template,
                debug_dir=temp / "debug",
            ).transform_prompt(
                scene_number=7,
                original_prompt="idea",
                width=1280,
                height=704,
            )

            self.assertEqual("S", (temp / "debug" / "scene_0007_system.txt").read_text(encoding="utf-8"))
            self.assertEqual("U 1280 704 idea", (temp / "debug" / "scene_0007_user.txt").read_text(encoding="utf-8"))
            self.assertEqual("{not json but ok}", (temp / "debug" / "scene_0007_response.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
