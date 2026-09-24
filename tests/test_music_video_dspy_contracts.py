import unittest
from contextlib import nullcontext

from feverslop.prompting.guide_loader import load_markdown_guide
from feverslop.prompting.music_video_modules import MusicVideoPromptModules
from feverslop.prompting.music_video_signatures import (
    build_music_video_signature_bundle,
)


class MusicVideoDspyContractTests(unittest.TestCase):
    def test_all_creative_predictors_receive_compact_non_mutating_scene_inputs(self):
        import json
        from copy import deepcopy
        calls = []
        modules = object.__new__(MusicVideoPromptModules)
        def predict(**kwargs):
            calls.append(kwargs)
            return {"concepts": {}, "detail": "detail"}
        modules._predictors = {name: predict for name in ("concept_map", "repair_concepts", "detail")}
        modules._context = lambda **kwargs: nullcontext()
        modules._lm = object()
        scene = dict(segment_id="s1", lyrics="Keep all lyrics", start=10, end=12,
                     performance_intervals=[{"vocal_events": [{"alignment": "x" * 800000}]}],
                     word_timestamps=[{"word": "Keep"}], actors=[{"id": "lead", "role": "singer"}])
        original = deepcopy(scene)
        modules.concepts({"CURRENT_BATCH_SEGMENTS": [scene]}, batch=True)
        modules.concepts({"SEGMENT_TIMELINE_JSON": [scene]})
        modules.repair_concepts({"MISSING_SEGMENTS": [scene]})
        modules.detail("camera", {"scene": scene}, "Preserve composition")
        for call in calls:
            payload = json.dumps(call["payload"])
            self.assertLess(len(payload), 300)
            self.assertIn("Keep all lyrics", payload)
            self.assertIn("singer", payload)
            self.assertNotIn("performance_intervals", payload)
        self.assertEqual(original, scene)

    def test_signature_bundle_covers_each_classic_request_shape(self):
        bundle = build_music_video_signature_bundle()

        self.assertEqual(
            {"story_idea", "style_block", "subject_locations", "narrative_contract", "narrative_milestone_bindings", "concept_map", "detail", "t2i", "i2v", "summary", "repair_concepts"},
            set(bundle),
        )
        self.assertIn("guide", bundle["story_idea"].input_fields)
        self.assertIn("result", bundle["subject_locations"].output_fields)
        self.assertIn("concepts", bundle["concept_map"].output_fields)
        self.assertIn("locations", bundle["narrative_contract"].input_fields)
        self.assertIn("actors", bundle["narrative_contract"].input_fields)
        self.assertIn("contract", bundle["narrative_contract"].output_fields)

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
        self.assertEqual(2048, calls[0]["config"]["max_tokens"])

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

        self.assertEqual(4096, calls[0]["config"]["max_tokens"])

    def test_concept_requests_use_a_bounded_lm_when_the_runtime_supports_it(self):
        lm_calls = []
        context_lms = []

        class Predictor:
            def __call__(self, **_kwargs):
                return {"concepts": {}}

        class LLM:
            model = "fake-model"
            client = object()

        class Runtime:
            def make_lm(self, _llm, *, max_tokens=None, task=None):
                lm_calls.append((max_tokens, task))
                return f"lm-{max_tokens or 'default'}"

            @staticmethod
            def context(*, lm):
                context_lms.append(lm)
                return nullcontext()

            predict = staticmethod(lambda _signature: Predictor())

        modules = MusicVideoPromptModules(LLM(), dspy_runtime=Runtime())
        modules.concepts({"CURRENT_BATCH_SEGMENTS": [{"segment_id": "segment_001"}]}, batch=True)

        self.assertIn((4096, "planner"), lm_calls)
        self.assertEqual("lm-4096", context_lms[-1])

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

        self.assertEqual(4096, calls[0]["config"]["max_tokens"])

    def test_repair_concepts_scale_output_limit_per_expected_key(self):
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

        modules.repair_concepts({"EXPECTED_KEYS": ["segment_005"]})
        modules.repair_concepts({"EXPECTED_KEYS": ["segment_005", "segment_008"]})

        self.assertEqual(1792, calls[0]["config"]["max_tokens"])
        self.assertEqual(2560, calls[1]["config"]["max_tokens"])

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

    def test_concept_and_repair_guides_require_semantic_scene_state(self):
        for name in ("music-video-concepts", "music-video-concept-repair"):
            with self.subTest(name=name):
                guide = load_markdown_guide(name).lower()
                self.assertIn('"narrative"', guide)
                self.assertIn('"story_beat"', guide)
                self.assertIn('"objective"', guide)
                self.assertIn('"action_phase"', guide)
                self.assertIn('"milestones"', guide)
                self.assertIn('"cast_states"', guide)
                self.assertIn('"props"', guide)
                self.assertIn('"reset_events"', guide)
                self.assertIn('"causal_events"', guide)
                self.assertIn('"incoming"', guide)
                self.assertIn('"outgoing"', guide)
                self.assertIn('"transition_events"', guide)
                self.assertIn('"transition_from_previous"', guide)
                self.assertIn("terminally absent", guide)
                self.assertIn("one-shot", guide)
                self.assertIn("explicit causal reset", guide)
                self.assertIn("chronology_exceptions", guide)

    def test_all_classic_guides_are_package_resources(self):
        for name in (
            "music-video-story-idea", "music-video-style", "music-video-subject-locations",
            "music-video-concepts", "music-video-concept-repair", "music-video-summary",
            "music-video-detail", "music-video-t2i", "music-video-i2v",
        ):
            self.assertTrue(load_markdown_guide(name).strip())

    def test_subject_locations_signature_declares_cast_idea(self):
        # A supplied cast_idea must reach the model instead of being dropped
        # with an "Input contains fields not in signature" warning.
        bundle = build_music_video_signature_bundle()

        self.assertIn("cast_idea", bundle["subject_locations"].input_fields)

    def test_subject_locations_call_forwards_cast_idea_and_effective_budget(self):
        calls = []

        class Predictor:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                return {"result": {"subject": "", "actors": [], "locations": []}}

        class LLM:
            model = "fake-model"
            client = object()

        class Runtime:
            def make_lm(self, llm):
                return "lm"

            context = staticmethod(lambda **kwargs: nullcontext())
            predict = staticmethod(lambda signature: Predictor())

        modules = MusicVideoPromptModules(LLM(), dspy_runtime=Runtime())

        modules.subject_locations("A duo performs.", "configured anchors", "Two singers, one bassist.")

        self.assertEqual("Two singers, one bassist.", calls[0]["cast_idea"])
        self.assertEqual("A duo performs.", calls[0]["story_idea"])
        self.assertEqual(2048, calls[0]["config"]["max_tokens"])

    def test_repair_and_subject_guides_document_the_wired_contract(self):
        repair_guide = load_markdown_guide("music-video-concept-repair").lower()
        self.assertIn("expected_keys", repair_guide)
        self.assertIn("neighbor_distance", repair_guide)
        self.assertIn("prior_segment_state", repair_guide)

        subject_guide = load_markdown_guide("music-video-subject-locations").lower()
        self.assertIn("cast idea", subject_guide)


if __name__ == "__main__":
    unittest.main()
