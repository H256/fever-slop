import tempfile
import unittest
from pathlib import Path


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def complete_prompt(self, system_prompt: str, prompt: str) -> str:
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        return self.response


class StoryboardPromptTransformerTests(unittest.TestCase):
    def test_template_transformer_passes_system_and_filled_user_prompt_to_llm(self):
        from feverslop.prompting.storyboard_prompt_transformer import TemplateStoryboardPromptTransformer

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            template = temp / "template.md"
            template.write_text(
                "[META]\nignored\n\n[SYSTEM]\nSystem rules\n\n[USER]\n"
                "TARGET IMAGE ASPECT RATIO: {{width}}:{{height}} (width:height).\n"
                "User idea: {{original_prompt}}\n",
                encoding="utf-8",
            )
            llm = FakeLLM(" raw non-json response \n")

            result = TemplateStoryboardPromptTransformer(
                llm=llm,
                template_path=template,
                debug_dir=temp / "debug",
            ).transform_prompt(
                scene_number=1,
                original_prompt="cinematic frame",
                width=1920,
                height=1088,
            )

        self.assertEqual("raw non-json response", result)
        self.assertEqual("System rules", llm.calls[0]["system_prompt"])
        self.assertEqual(
            "TARGET IMAGE ASPECT RATIO: 1920:1088 (width:height).\n"
            "User idea: cinematic frame",
            llm.calls[0]["prompt"],
        )

    def test_template_transformer_writes_debug_files(self):
        from feverslop.prompting.storyboard_prompt_transformer import TemplateStoryboardPromptTransformer

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            template = temp / "template.md"
            template.write_text("[SYSTEM]\nS\n\n[USER]\nU {{width}} {{height}} {{original_prompt}}", encoding="utf-8")

            TemplateStoryboardPromptTransformer(
                llm=FakeLLM("{not json but ok}"),
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
