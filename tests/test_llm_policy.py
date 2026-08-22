import unittest


class LLMPolicyTests(unittest.TestCase):
    def test_short_structured_tasks_get_bounded_default_budget(self):
        from feverslop.prompting.llm_policy import policy_for

        policy = policy_for("lyric_alignment")

        self.assertEqual("structured", policy.profile)
        self.assertEqual(512, policy.max_tokens)

    def test_long_form_music_video_tasks_use_creative_budget(self):
        from feverslop.prompting.llm_policy import policy_for

        for name in ("story_idea", "style_block"):
            policy = policy_for(name)
            self.assertEqual("creative", policy.profile, name)
            self.assertEqual(2048, policy.max_tokens, name)

    def test_batched_tasks_are_budgeted_by_call_site_multipliers(self):
        from feverslop.prompting.llm_policy import BATCHED_TASK_NAMES

        self.assertIn("concept_map", BATCHED_TASK_NAMES)
        self.assertIn("lyric_alignment", BATCHED_TASK_NAMES)

    def test_signature_bundle_task_names_have_explicit_policies(self):
        from feverslop.prompting.general_signatures import build_general_signature_bundle
        from feverslop.prompting.llm_policy import BATCHED_TASK_NAMES, known_task_names
        from feverslop.prompting.music_video_signatures import build_music_video_signature_bundle

        known = known_task_names()
        bundles = (
            build_music_video_signature_bundle(),
            build_general_signature_bundle(),
        )
        for bundle in bundles:
            for name in bundle:
                if name in BATCHED_TASK_NAMES:
                    continue
                self.assertIn(name, known)

    def test_creative_tasks_have_a_larger_budget(self):
        from feverslop.prompting.llm_policy import policy_for

        self.assertEqual("creative", policy_for("song_brief").profile)
        self.assertGreater(policy_for("song_brief").max_tokens, policy_for("lyric_alignment").max_tokens)

    def test_unknown_tasks_use_structured_safe_default(self):
        from feverslop.prompting.llm_policy import policy_for

        self.assertEqual("structured", policy_for("new_signature").profile)

    def test_multi_item_output_limits_scale_with_item_count(self):
        from feverslop.prompting.llm_policy import (
            concept_batch_max_tokens,
            lyric_alignment_max_tokens,
            msr_segments_max_tokens,
        )

        self.assertEqual(6144, concept_batch_max_tokens(10))
        self.assertEqual(4352, lyric_alignment_max_tokens(13))
        self.assertEqual(3072, msr_segments_max_tokens(4))

    def test_prompt_result_allows_scene_layer_to_handle_overlength_text(self):
        from feverslop.prompting.general_signatures import PromptResult

        result = PromptResult(prompt="word " * 151)

        self.assertEqual(151, len(result.prompt.split()))

    def test_storyboard_prompt_result_keeps_its_separate_safety_limit(self):
        from feverslop.prompting.general_signatures import StoryboardPromptResult

        with self.assertRaises(ValueError):
            StoryboardPromptResult(prompt="word " * 151)


if __name__ == "__main__":
    unittest.main()
