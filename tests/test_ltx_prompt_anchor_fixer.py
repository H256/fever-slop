import unittest

from feverslop.prompting.ltx_prompt_anchor_fixer import LTXPromptAnchorFixer


class LTXPromptAnchorFixerTests(unittest.TestCase):
    def test_fixes_single_prompt_from_startframe_prompt(self):
        fixer = LTXPromptAnchorFixer(subject_anchor="old druid shaman man")
        plan = [
            {
                "scene": 1,
                "z_image": {
                    "prompt": "A cinematic image of an old druid shaman man kneeling among moss and roots."
                },
                "ltx": {
                    "base_prompt": "The old druid shaman man stands beside a gnarled tree.",
                    "original_style_i2v_prompt": "The old druid shaman man stands beside a gnarled tree and shudders violently.",
                    "prompt_relay": [],
                },
                "metadata": {
                    "type": "instrumental",
                    "camera_motion": "locked low angle",
                    "character_motion": "calm breathing",
                    "base_concept": "forest ritual",
                },
            }
        ]

        fixed = fixer.fix_render_plan(plan)
        prompt = fixed[0]["ltx"]["original_style_i2v_prompt"].lower()

        self.assertIn("kneeling among moss and roots", prompt)
        self.assertIn("lock the first frame", prompt)
        self.assertIn("without fades", prompt)
        self.assertNotIn("animate the provided start frame", prompt)
        self.assertNotIn("stands beside a gnarled tree", prompt)

    def test_fixes_explicit_i2v_prompt_from_t2i_field(self):
        fixer = LTXPromptAnchorFixer(subject_anchor="old druid shaman man")
        plan = [
            {
                "scene": 1,
                "z_image": {
                    "prompt": "A cinematic image of an old druid shaman man kneeling among moss and roots."
                },
                "ltx": {
                    "base_prompt": "The old druid shaman man stands beside a gnarled tree.",
                    "original_style_i2v_prompt": "The old druid shaman man stands beside a gnarled tree and shudders violently.",
                    "i2v_prompt_from_t2i": "The old druid shaman man stands beside a gnarled tree and shudders violently.",
                    "prompt_relay": [],
                },
                "metadata": {
                    "type": "instrumental",
                    "camera_motion": "locked low angle",
                    "character_motion": "calm breathing",
                    "base_concept": "forest ritual",
                },
            }
        ]

        fixed = fixer.fix_render_plan(plan)
        prompt = fixed[0]["ltx"]["i2v_prompt_from_t2i"].lower()

        self.assertIn("kneeling among moss and roots", prompt)
        self.assertNotIn("stands beside a gnarled tree", prompt)


if __name__ == "__main__":
    unittest.main()
