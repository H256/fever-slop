import unittest
from contextlib import nullcontext

from feverslop.prompting.guide_loader import load_markdown_guide
from feverslop.prompting.music_video_modules import MusicVideoPromptModules
from feverslop.prompting.music_video_signatures import build_music_video_signature_bundle


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

    def test_injected_predictor_receives_markdown_guide_and_structured_data(self):
        calls = []

        class Predictor:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                return {"concepts": {"segment_001": "A forest path."}}

        class LLM:
            model = "fake-model"
            client = object()

        class Runtime:
            def make_lm(self, llm):
                return "lm"

            context = staticmethod(lambda **kwargs: nullcontext())
            predict = staticmethod(lambda signature: Predictor())

        modules = MusicVideoPromptModules(LLM(), dspy_runtime=Runtime())

        result = modules.concepts(
            {"GLOBAL_CONTEXT": {"location_constraint": "forest"}, "CURRENT_BATCH_SEGMENTS": [{"segment_id": "segment_001"}]},
        )

        self.assertEqual({"segment_001": "A forest path."}, result)
        self.assertIn("location_constraint", calls[0]["guide"])
        self.assertIn("CURRENT_BATCH_SEGMENTS", calls[0]["payload"])
        self.assertEqual(512, calls[0]["config"]["max_tokens"])

    def test_batched_concepts_scale_output_limit_per_segment(self):
        calls = []

        class Predictor:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                return {"concepts": {}}

        class LLM:
            model = "fake-model"
            client = object()

        class Runtime:
            def make_lm(self, llm):
                return "lm"

            context = staticmethod(lambda **kwargs: nullcontext())
            predict = staticmethod(lambda signature: Predictor())

        modules = MusicVideoPromptModules(LLM(), dspy_runtime=Runtime())

        modules.concepts(
            {"CURRENT_BATCH_SEGMENTS": [{"segment_id": f"segment_{i:03}"} for i in range(1, 11)]},
            batch=True,
        )

        self.assertEqual(6144, calls[0]["config"]["max_tokens"])

    def test_legacy_concepts_scale_output_limit_by_timeline_size(self):
        calls = []

        class Predictor:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                return {"concepts": {}}

        class LLM:
            model = "fake-model"
            client = object()

        class Runtime:
            def make_lm(self, llm):
                return "lm"

            context = staticmethod(lambda **kwargs: nullcontext())
            predict = staticmethod(lambda signature: Predictor())

        modules = MusicVideoPromptModules(LLM(), dspy_runtime=Runtime())
        modules.concepts({"SEGMENT_TIMELINE_JSON": [{"segment_id": str(i)} for i in range(10)]})

        self.assertEqual(6144, calls[0]["config"]["max_tokens"])

    def test_dspy_predictor_receives_caller_timeout_as_lm_config(self):
        calls = []

        class Predictor:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                return {"concepts": {"segment_001": "A forest path."}}

        class LLM:
            model = "fake-model"
            client = object()

        class Runtime:
            def make_lm(self, llm):
                return "lm"

            context = staticmethod(lambda **kwargs: nullcontext())
            predict = staticmethod(lambda signature: Predictor())

        modules = MusicVideoPromptModules(LLM(), dspy_runtime=Runtime())

        modules.concepts({"CURRENT_BATCH_SEGMENTS": []}, timeout=42.0)

        self.assertEqual(42.0, calls[0]["config"]["timeout"])

    def test_classic_concept_guide_requires_standalone_concrete_continuity(self):
        guide = load_markdown_guide("music-video-concepts").lower()

        self.assertIn("each concept must stand alone", guide)
        self.assertIn("repeat key visible continuity details", guide)
        self.assertIn("do not invent or assume character details", guide)
        self.assertIn('never write "the same character"', guide)
        self.assertIn('"from earlier"', guide)
        self.assertIn("every available performer", guide)
        self.assertIn("role-defining instrument", guide)

    def test_all_classic_guides_are_package_resources(self):
        for name in (
            "music-video-story-idea", "music-video-style", "music-video-subject-locations",
            "music-video-concepts", "music-video-concept-repair", "music-video-summary",
            "music-video-detail", "music-video-t2i", "music-video-i2v",
        ):
            self.assertTrue(load_markdown_guide(name).strip())


if __name__ == "__main__":
    unittest.main()
