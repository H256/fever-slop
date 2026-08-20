from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path

import dspy

from feverslop.application.ingredients_vision_prompt import build_ingredients_vision_prompt
from feverslop.domain.vision_references import ReferenceImage
from feverslop.prompting.guide_loader import load_markdown_guide
from feverslop.prompting.ingredients_modules import IngredientsPromptModules
from feverslop.prompting.ingredients_signatures import build_ingredients_signature_bundle
from tests.fakellm import FakeVisionLLM, FailingVisionLLM


class IngredientsVisionPromptTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.actor = Path(self.temp_dir.name) / "actor.png"
        self.location = Path(self.temp_dir.name) / "location.png"
        self.actor.write_bytes(b"actor")
        self.location.write_bytes(b"location")
        self.references = [
            ReferenceImage("mara", "actor", self.actor),
            ReferenceImage("archive", "location", self.location),
        ]
        self.reference_metadata = [
            {
                "id": "mara",
                "type": "actor",
                "t2i_description": "Mara has cropped black hair and a charcoal coat.",
                "manifest_description": "A guarded archivist with a silver ear cuff.",
            },
            {
                "id": "archive",
                "type": "location",
                "t2i_description": "A narrow stone archive lined with shelves.",
                "manifest_description": "Dusty subterranean records room.",
            },
        ]
        self.target_context = {
            "action": "Mara searches the shelves while the lights fail.",
            "camera": "Slow push from the doorway to a close medium shot.",
            "acting": "Controlled fear becomes resolve.",
            "dialogue": "I know you are here.",
            "continuity": "Her coat is wet from the previous shot.",
            "duration": "8 seconds",
        }
        self.fallback_reference = "fallback refs"
        self.fallback_invariants = "fallback shot invariants"

    @staticmethod
    def shot_invariants() -> str:
        return " ".join(f"invariant{i}" for i in range(80))

    def build(self, llm):
        if llm is not None:
            llm.model = "fake-model"
            llm.client = object()
        calls = []

        class Predictor:
            def __call__(predictor_self, **kwargs):
                calls.append(kwargs)
                if isinstance(llm, FailingVisionLLM):
                    raise RuntimeError("vision failed")
                response = getattr(llm, "_canned", "")
                try:
                    response = json.loads(response)
                except (TypeError, json.JSONDecodeError):
                    pass
                return {"result": response}

        class Runtime:
            def make_lm(self, llm):
                return "lm"

            context = staticmethod(lambda **kwargs: nullcontext())
            predict = staticmethod(lambda signature: Predictor())

        self.last_module_calls = calls
        return build_ingredients_vision_prompt(
            llm=llm,
            references=self.references,
            reference_metadata=self.reference_metadata,
            target_context=self.target_context,
            fallback_reference_description=self.fallback_reference,
            fallback_shot_invariants=self.fallback_invariants,
            dspy_runtime=Runtime(),
        )

    def test_signature_and_module_pass_typed_payload_and_images(self):
        calls = []

        class Predictor:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                return {"result": {"references": [], "shot_invariants": " ".join(f"invariant{i}" for i in range(80))}}

        class LLM:
            model = "fake-model"
            client = object()

        class Runtime:
            def make_lm(self, llm):
                return "lm"

            context = staticmethod(lambda **kwargs: nullcontext())
            predict = staticmethod(lambda signature: Predictor())

        bundle = build_ingredients_signature_bundle()
        self.assertIn("images", bundle["vision"].input_fields)
        self.assertEqual(list[dspy.Image], bundle["vision"].input_fields["images"].annotation)
        self.assertIn("result", bundle["vision"].output_fields)
        result_type = bundle["vision"].output_fields["result"].annotation
        self.assertIn("t2i_description", result_type.model_fields["references"].annotation.__args__[0].model_fields)
        self.assertIn("t2i_description", bundle["vision"].input_fields["payload"].annotation.model_fields["references"].annotation.__args__[0].model_fields)
        self.assertIn("Ingredients", load_markdown_guide("ingredients-vision"))
        modules = IngredientsPromptModules(LLM(), dspy_runtime=Runtime())
        modules.vision(
            {"references": self.reference_metadata, "target_context": self.target_context},
            [self.actor, self.location],
        )
        self.assertEqual("mara", calls[0]["payload"].references[0].id)
        self.assertEqual("Mara has cropped black hair and a charcoal coat.", calls[0]["payload"].references[0].t2i_description)
        self.assertEqual(self.target_context, calls[0]["payload"].target_context)
        self.assertTrue(all(isinstance(image, dspy.Image) for image in calls[0]["images"]))
        self.assertTrue(all(image.url.startswith("data:image/") for image in calls[0]["images"]))

    def test_skips_vision_request_when_endpoint_reports_no_vision(self):
        class LLM:
            model = "text-only"
            client = object()

            @staticmethod
            def model_supports_vision():
                return False

        result = self.build(LLM())

        self.assertEqual("vision unavailable", result.fallback_reason)
        self.assertEqual([], self.last_module_calls)

    def test_missing_reference_and_word_boundaries_are_invalid_typed_outputs(self):
        class Predictor:
            def __init__(self, result):
                self.result = result

            def __call__(self, **kwargs):
                return {"result": self.result}

        class LLM:
            model = "fake-model"
            client = object()

        class Runtime:
            def make_lm(self, llm):
                return "lm"

            context = staticmethod(lambda **kwargs: nullcontext())

            def predict(self, signature):
                return Predictor({"references": [], "shot_invariants": "too short"})

        result = build_ingredients_vision_prompt(
            llm=LLM(),
            references=self.references,
            reference_metadata=self.reference_metadata,
            target_context=self.target_context,
            fallback_reference_description=self.fallback_reference,
            fallback_shot_invariants=self.fallback_invariants,
            dspy_runtime=Runtime(),
        )
        self.assertEqual("invalid response", result.fallback_reason)

    def test_builds_grounded_prompt_and_attaches_references_in_source_order(self):
        llm = FakeVisionLLM(
            json.dumps({
                "references": [
                    {"id": "mara", "type": "actor", "t2i_description": "Cropped black hair, silver ear cuff, charcoal coat."},
                    {"id": "archive", "type": "location", "t2i_description": "Narrow stone archive with dusty shelves."},
                ],
                "shot_invariants": self.shot_invariants(),
            })
        )

        result = self.build(llm)

        call = self.last_module_calls[0]
        self.assertEqual(2, len(call["images"]))
        self.assertTrue(all(isinstance(image, dspy.Image) for image in call["images"]))
        self.assertFalse(any(isinstance(image, (Path, str)) for image in call["images"]))
        payload = call["payload"]
        self.assertEqual("mara", payload.references[0].id)
        self.assertEqual("archive", payload.references[1].id)
        self.assertEqual("Mara has cropped black hair and a charcoal coat.", payload.references[0].t2i_description)
        for key, value in self.target_context.items():
            self.assertEqual(value, payload.target_context[key])
        guide = str(call["guide"])
        self.assertIn("60-160", guide)
        self.assertIn("non-temporal", guide)
        self.assertIn("Do not schedule", guide)
        self.assertIn("singing", guide)
        self.assertIn("lip-sync", guide)
        self.assertIn("single continuous full-frame shot", guide)
        self.assertIn("do not reproduce", guide.lower())
        self.assertIn("framing", guide.lower())
        self.assertIn("layout", guide.lower())
        self.assertEqual(
            ["### Reference Sheet Description", "### Target Description"],
            [line for line in result.positive_prompt.splitlines() if line.startswith("###")],
        )
        for reference_id in ("mara", "archive"):
            self.assertIn(reference_id, result.reference_description)
        self.assertEqual(self.shot_invariants(), result.shot_invariants)

    def test_invalid_responses_use_deterministic_fallback_with_headings(self):
        invalid_responses = {
            "invalid json": "not JSON",
            "missing reference": {
                "references": [{"id": "mara", "type": "actor", "t2i_description": "Mara"}],
                "shot_invariants": self.shot_invariants(),
            },
            "wrong type": {
                "references": [
                    {"id": "mara", "type": "location", "t2i_description": "Mara"},
                    {"id": "archive", "type": "location", "t2i_description": "Archive"},
                ],
                "shot_invariants": self.shot_invariants(),
            },
            "empty target": {
                "references": [
                    {"id": "mara", "type": "actor", "t2i_description": "Mara"},
                    {"id": "archive", "type": "location", "t2i_description": "Archive"},
                ],
                "shot_invariants": "",
            },
            "short target": {
                "references": [
                    {"id": "mara", "type": "actor", "t2i_description": "Mara"},
                    {"id": "archive", "type": "location", "t2i_description": "Archive"},
                ],
                "shot_invariants": "too short",
            },
        }
        for label, response in invalid_responses.items():
            with self.subTest(label):
                canned = json.dumps(response) if isinstance(response, dict) else response
                result = self.build(FakeVisionLLM(canned))
                self.assertEqual(self.fallback_reference, result.reference_description)
                self.assertEqual(self.fallback_invariants, result.shot_invariants)
                self.assertEqual("invalid response", result.fallback_reason)
                self.assertEqual(
                    "### Reference Sheet Description\nfallback refs\n\n### Target Description\nfallback shot invariants",
                    result.positive_prompt,
                )

    def test_model_exception_preserves_vision_unavailable_reason(self):
        for llm in (FailingVisionLLM(), None):
            with self.subTest(llm=llm):
                result = self.build(llm)
                self.assertEqual(self.fallback_reference, result.reference_description)
                self.assertEqual(self.fallback_invariants, result.shot_invariants)
                self.assertEqual("vision unavailable", result.fallback_reason)

    def test_probe_failure_reports_probe_failed_reason(self):
        test_case = self

        class LLM:
            model = "fake-model"
            client = object()

            def model_supports_vision(self):
                raise ConnectionError("probe outage")

            def complete_prompt_with_images(self, *args, **kwargs):
                test_case.last_module_calls.append({"unexpected_vision_call": True})
                raise AssertionError("vision call attempted despite failing probe")

        with self.assertLogs(
            "feverslop.application.ingredients_vision_prompt", level="WARNING"
        ) as logs:
            result = self.build(LLM())

        self.assertEqual("vision probe failed", result.fallback_reason)
        self.assertEqual([], self.last_module_calls)
        self.assertTrue(any("ConnectionError" in line for line in logs.output))

    def test_probe_true_proceeds_to_vision_call(self):
        llm = FakeVisionLLM(
            json.dumps({
                "references": [
                    {"id": "mara", "type": "actor", "t2i_description": "Cropped black hair, silver ear cuff, charcoal coat."},
                    {"id": "archive", "type": "location", "t2i_description": "Narrow stone archive with dusty shelves."},
                ],
                "shot_invariants": self.shot_invariants(),
            })
        )
        llm.model_supports_vision = lambda: True

        result = self.build(llm)

        self.assertEqual(1, len(self.last_module_calls))
        self.assertIsNone(result.fallback_reason)


if __name__ == "__main__":
    unittest.main()
