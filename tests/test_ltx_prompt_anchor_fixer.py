import unittest

from ltx_prompt_anchor_fixer import LTXPromptAnchorFixer


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
        self.assertIn("preserve the exact startframe composition", prompt)
        self.assertNotIn("stands beside a gnarled tree", prompt)


if __name__ == "__main__":
    unittest.main()
