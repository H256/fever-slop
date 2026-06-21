import unittest

from feverslop.prompting.music_video_prompt_style import (
    build_location_constraint,
    build_concept_mapper_system_prompt,
    build_detail_system_prompt,
    build_i2v_system_prompt,
    build_t2i_system_prompt,
    build_video_payload,
    performance_policy,
)


class PerformancePolicyTests(unittest.TestCase):
    def test_vocal_policy_requires_singing_and_lip_sync(self):
        policy = performance_policy("vocals").lower()

        self.assertIn("singing with passion", policy)
        self.assertIn("lip sync", policy)
        self.assertNotIn("closed or relaxed mouth", policy)

    def test_instrumental_policy_blocks_singing_and_mouth_motion(self):
        policy = performance_policy("instrumental").lower()

        self.assertIn("must not sing", policy)
        self.assertIn("no lip sync", policy)
        self.assertIn("closed or relaxed mouth", policy)
        self.assertNotIn("singing with passion", policy)

    def test_mixed_policy_allows_relay_without_forcing_full_scene_singing(self):
        policy = performance_policy("mixed").lower()

        self.assertIn("alternates", policy)
        self.assertIn("only during vocal intervals", policy)
        self.assertIn("silent intervals", policy)
        self.assertNotIn("throughout the shot", policy)


class PromptInstructionTests(unittest.TestCase):
    def test_location_constraint_requires_allowed_locations_when_configured(self):
        constraint = build_location_constraint([
            "A lush, ancient forest with dappled sunlight",
            "A secluded, crystal-clear spring in a Locus amoenus",
        ]).lower()

        self.assertIn("allowed locations", constraint)
        self.assertIn("every scene concept", constraint)
        self.assertIn("every z-image prompt", constraint)
        self.assertIn("must visibly take place", constraint)
        self.assertIn("do not invent other locations", constraint)
        self.assertIn("a lush, ancient forest with dappled sunlight", constraint)
        self.assertIn("a secluded, crystal-clear spring in a locus amoenus", constraint)

    def test_location_constraint_is_empty_without_locations(self):
        self.assertEqual("", build_location_constraint([]))

    def test_t2i_system_prompt_uses_reference_style_structure(self):
        prompt = build_t2i_system_prompt().lower()

        self.assertIn("current visual prompt as the main scene foundation", prompt)
        self.assertIn("a high resolution cinematic photograph of a", prompt)
        self.assertIn("do not use metaphors", prompt)
        self.assertIn("only send the final prompt text", prompt)
        self.assertIn("do not say the character is singing", prompt)
        self.assertIn("location_constraint", prompt)
        self.assertIn("every z-image prompt", prompt)

    def test_i2v_system_prompt_adapts_performance_policy(self):
        prompt = build_i2v_system_prompt("instrumental").lower()

        self.assertIn("convert the user's concept prompt into a dynamic image-to-video prompt", prompt)
        self.assertIn("keep the subject visible", prompt)
        self.assertIn("must not sing", prompt)
        self.assertIn("no lip sync", prompt)
        self.assertNotIn("subject must be physically singing", prompt)

    def test_i2v_system_prompt_relaxes_mixed_scene_singing_policy(self):
        prompt = build_i2v_system_prompt("mixed").lower()

        self.assertIn("vocal intervals", prompt)
        self.assertIn("silent intervals", prompt)
        self.assertIn("only during vocal intervals", prompt)
        self.assertNotIn("throughout the shot", prompt)

    def test_concept_mapper_prompt_requires_continuity_and_standalone_segments(self):
        prompt = build_concept_mapper_system_prompt(batch=True).lower()

        self.assertIn("continuous visual story", prompt)
        self.assertIn("each concept must stand alone", prompt)
        self.assertIn("prompt guidance", prompt)
        self.assertIn("location_constraint", prompt)
        self.assertIn("every scene concept", prompt)
        self.assertIn("shot types", prompt)
        self.assertIn("instrumental segments", prompt)
        self.assertIn("do not say the character is singing", prompt)

    def test_detail_prompt_is_label_specific(self):
        prompt = build_detail_system_prompt("Camera Motion", segment_type="vocals").lower()

        self.assertIn("output only camera movement phrases", prompt)
        self.assertIn("do not combine multiple categories", prompt)
        self.assertIn("singing with passion", prompt)

    def test_video_payload_includes_segment_timing_and_policies(self):
        payload = build_video_payload(
            segment={"segment_id": "segment_001", "type": "instrumental", "start": 1.0, "end": 4.0},
            concept="A figure waits in a rainlit station.",
            scene_details={"camera_motion": "slow push-in", "character_motion": "small posture shifts"},
            global_context={
                "subject": "a man with gray hair wearing a long black coat",
                "story_idea": "A quiet journey through an empty city.",
                "style": "cinematic realism",
                "locations": ["rainlit station"],
                "prompt_guidance": {"shot_types": "close-up", "lighting": "soft rim light"},
            },
            custom_instructions="Keep the coat consistent.",
        )

        self.assertEqual("instrumental", payload["performance_mode"])
        self.assertIn("must not sing", payload["performance_policy"].lower())
        self.assertEqual("slow push-in", payload["camera_motion"])
        self.assertEqual("segment_001", payload["segment"]["segment_id"])
        self.assertEqual("close-up", payload["prompt_guidance"]["shot_types"])


if __name__ == "__main__":
    unittest.main()
