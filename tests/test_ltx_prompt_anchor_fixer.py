import unittest

from feverslop.prompting.ltx_prompt_anchor_fixer import LTXPromptAnchorFixer


class LTXPromptAnchorFixerTests(unittest.TestCase):
    def test_fixes_single_prompt_from_startframe_prompt(self):
        fixer = LTXPromptAnchorFixer(subject_anchor="old druid shaman man")
        plan = [
            {
                "scene": 1,
                "z_image": {
                    "prompt": "A cinematic image of an old druid shaman man kneeling among moss and roots.",
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
            },
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
                    "prompt": "A cinematic image of an old druid shaman man kneeling among moss and roots.",
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
            },
        ]

        fixed = fixer.fix_render_plan(plan)
        prompt = fixed[0]["ltx"]["i2v_prompt_from_t2i"].lower()

        self.assertIn("kneeling among moss and roots", prompt)
        self.assertNotIn("stands beside a gnarled tree", prompt)

    def test_sentence_limit_keeps_complete_sentence(self):
        from feverslop.prompting.ltx_prompt_anchor_fixer import _sentence_limit
        text = "A. B. C. D. E. F. G. H. I. J. K. L. M. N. O. P. Q. R. S. T."
        result = _sentence_limit(text, max_chars=20)
        self.assertEqual("A. B. C. D. E.", result)

    def test_long_vocal_startframe_preserves_complete_lip_sync_instruction(self):
        fixer = LTXPromptAnchorFixer(subject_anchor="warrior_lead", max_base_prompt_chars=600)
        plan = [{
            "scene": 5,
            "z_image": {"prompt": "A detailed cathedral chamber. " * 80},
            "ltx": {"base_prompt": "The party approaches the gate.", "prompt_relay": []},
            "metadata": {
                "type": "vocals",
                "camera_motion": "slow tracking shot",
                "character_motion": "the party advances together",
                "base_concept": "the party prepares to fight",
            },
        }]

        prompt = fixer.fix_render_plan(plan)[0]["ltx"]["i2v_prompt_from_t2i"]

        self.assertLessEqual(len(prompt), 600)
        self.assertIn("warrior_lead remains clearly visible and sings with controlled lip sync", prompt)
        self.assertNotRegex(prompt, r"(?:Subject or|Story beat|Camera motion):?$")

    def test_uses_scene_actor_ids_instead_of_global_subject(self):
        fixer = LTXPromptAnchorFixer(subject_anchor="global warrior")
        plan = [{
            "scene": 7,
            "references": {"actor_ids": ["mage_lead", "rogue_lead"]},
            "z_image": {"prompt": "The mage and rogue enter the dungeon."},
            "ltx": {"base_prompt": "They approach the gate.", "prompt_relay": [{"state": "singing", "prompt": "sing"}]},
            "metadata": {"type": "vocals"},
        }]

        scene = fixer.fix_render_plan(plan)[0]

        self.assertIn("mage_lead and rogue_lead remain clearly visible and sing", scene["ltx"]["i2v_prompt_from_t2i"])
        self.assertIn("mage_lead and rogue_lead remain clearly visible", scene["ltx"]["prompt_relay"][0]["prompt"])
        self.assertNotIn("global warrior", scene["ltx"]["i2v_prompt_from_t2i"])


if __name__ == "__main__":
    unittest.main()
