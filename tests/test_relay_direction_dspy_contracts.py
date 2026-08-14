import unittest
from contextlib import nullcontext

from feverslop.prompting.guide_loader import load_markdown_guide
from feverslop.prompting.relay_direction_builder import RelayDirectionBuilder
from feverslop.prompting.relay_signatures import build_relay_signature_bundle


class RelayDirectionDspyContractTests(unittest.TestCase):
    def test_relay_contract_has_typed_output_and_editable_guide(self):
        bundle = build_relay_signature_bundle()

        self.assertIn("result", bundle.output_fields)
        guide = load_markdown_guide("relay-directions")
        self.assertIn("under the configured word limit", guide)
        self.assertIn("singing", guide)

    def test_relay_dspy_module_receives_structured_payload(self):
        calls = []

        class Predictor:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                return {"result": {"directions": [{"index": 0, "prompt": "Warrior turns."}]}}

        class Runtime:
            def make_lm(self, llm):
                return "lm"

            context = staticmethod(lambda **kwargs: nullcontext())
            predict = staticmethod(lambda signature: Predictor())

        class LLM:
            model = "fake-model"
            client = object()

        builder = RelayDirectionBuilder(LLM(), dspy_runtime=Runtime(), max_words=28)
        scene = {
            "metadata": {"type": "instrumental"},
            "ltx": {"base_prompt": "A dark road."},
            "prompt_relay": [{"frame_start": 0, "frame_end": 9, "state": "instrumental", "prompt": "turns"}],
        }
        builder.compact_scene_relays(scene, scene["prompt_relay"])

        self.assertEqual("instrumental", calls[0]["payload"]["scene_type"])
        self.assertEqual(28, calls[0]["max_words"])
        self.assertNotIsInstance(calls[0]["payload"], str)

    def test_typed_relay_output_is_truncated_to_max_words(self):
        class Predictor:
            def __call__(self, **kwargs):
                return {"result": {"directions": [{"index": 0, "prompt": "one two three four five six"}]}}

        class Runtime:
            def make_lm(self, llm):
                return "lm"

            context = staticmethod(lambda **kwargs: nullcontext())
            predict = staticmethod(lambda signature: Predictor())

        class LLM:
            model = "fake-model"
            client = object()

        builder = RelayDirectionBuilder(LLM(), dspy_runtime=Runtime(), max_words=4)
        scene = {
            "metadata": {"type": "instrumental"},
            "ltx": {"base_prompt": "A dark road."},
            "prompt_relay": [{"frame_start": 0, "frame_end": 9, "state": "instrumental"}],
        }

        prompt = builder.compact_scene_relays(scene, scene["prompt_relay"])[0]["prompt"]

        self.assertEqual(4, len(prompt.split()))
        self.assertEqual("one two three four", prompt)

    def test_relay_without_dspy_uses_deterministic_fallback_without_text_completion(self):
        class LLM:
            def complete_prompt(self, **kwargs):
                raise AssertionError("legacy text completion must not be used")

        builder = RelayDirectionBuilder(LLM())
        scene = {
            "metadata": {"type": "instrumental", "character_motion": "The warrior turns."},
            "ltx": {"base_prompt": "A dark road."},
            "prompt_relay": [{"frame_start": 0, "frame_end": 9, "state": "instrumental"}],
        }

        prompt = builder.compact_scene_relays(scene, scene["prompt_relay"])[0]["prompt"]

        self.assertIn("preserve the same shot", prompt)

    def test_deterministic_singing_fallback_respects_max_words_and_constraints(self):
        class LLM:
            pass

        builder = RelayDirectionBuilder(LLM(), max_words=28)
        scene = {
            "metadata": {
                "type": "vocals",
                "camera_motion": "the camera drifts slowly",
                "character_motion": "the warrior raises his hand",
            },
            "ltx": {"base_prompt": "A dark road."},
            "prompt_relay": [{"frame_start": 0, "frame_end": 9, "state": "singing"}],
        }

        prompt = builder.compact_scene_relays(scene, scene["prompt_relay"])[0]["prompt"]

        self.assertTrue(prompt)
        self.assertLessEqual(len(prompt.split()), 28)
        self.assertIn("singing", prompt)
        self.assertIn("lip sync", prompt)

    def test_safety_repair_respects_max_words_and_preserves_singing_constraints(self):
        class Predictor:
            def __call__(self, **kwargs):
                return {"result": {"directions": [{"index": 0, "prompt": "tree sings"}]}}

        class Runtime:
            def make_lm(self, llm):
                return "lm"

            context = staticmethod(lambda **kwargs: nullcontext())
            predict = staticmethod(lambda signature: Predictor())

        class LLM:
            model = "fake-model"
            client = object()

        builder = RelayDirectionBuilder(LLM(), dspy_runtime=Runtime(), max_words=28)
        scene = {
            "metadata": {"type": "vocals"},
            "ltx": {"base_prompt": "A dark road."},
            "prompt_relay": [{"frame_start": 0, "frame_end": 9, "state": "singing"}],
        }

        prompt = builder.compact_scene_relays(scene, scene["prompt_relay"])[0]["prompt"]

        self.assertTrue(prompt)
        self.assertLessEqual(len(prompt.split()), 28)
        self.assertIn("singing", prompt)
        self.assertIn("lip sync", prompt)


if __name__ == "__main__":
    unittest.main()
