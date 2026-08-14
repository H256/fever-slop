import unittest

from feverslop.prompting.guide_loader import load_markdown_guide
from feverslop.prompting.music_video_modules import MusicVideoPromptModules
from feverslop.prompting.music_video_signatures import build_music_video_signature_bundle
from tests.fakellm import FakeLLM


class MusicVideoDspyContractTests(unittest.TestCase):
    def test_signature_bundle_covers_each_classic_request_shape(self):
        bundle = build_music_video_signature_bundle()

        self.assertEqual(
            {"story_idea", "style_block", "subject_locations", "concept_map", "detail", "t2i", "i2v", "summary", "repair_concepts"},
            set(bundle),
        )
        self.assertIn("guide", bundle["story_idea"].input_fields)
        self.assertIn("result", bundle["subject_locations"].output_fields)
        self.assertIn("concepts", bundle["concept_map"].output_fields)

    def test_legacy_test_transport_receives_markdown_guide_and_structured_data(self):
        llm = FakeLLM('{"segment_001": "A forest path."}')
        modules = MusicVideoPromptModules(llm)

        result = modules.concepts(
            {"GLOBAL_CONTEXT": {"location_constraint": "forest"}, "CURRENT_BATCH_SEGMENTS": [{"segment_id": "segment_001"}]},
        )

        self.assertEqual('{"segment_001": "A forest path."}', result)
        self.assertIn("location_constraint", llm.calls[0].system_prompt)
        self.assertIn("CURRENT_BATCH_SEGMENTS", llm.calls[0].prompt)
        self.assertNotIn("{GLOBAL_CONTEXT}", llm.calls[0].system_prompt)

    def test_all_classic_guides_are_package_resources(self):
        for name in (
            "music-video-story-idea", "music-video-style", "music-video-subject-locations",
            "music-video-concepts", "music-video-concept-repair", "music-video-summary",
            "music-video-detail", "music-video-t2i", "music-video-i2v",
        ):
            self.assertTrue(load_markdown_guide(name).strip())


if __name__ == "__main__":
    unittest.main()
