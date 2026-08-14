import unittest
from contextlib import nullcontext

from feverslop.prompting.dspy_h3_generator_core import VideoPromptGenerator
from feverslop.prompting.dspy_h3_models import (
    BaseVideoPrompt,
    MusicIntent,
    PromptPlan,
    ResolvedPromptPlan,
    ResolvedReference,
)
from feverslop.prompting.dspy_runtime import DspyRuntime, H3SignatureBundle


class DspyRuntimeTests(unittest.TestCase):
    def test_make_lm_uses_injected_factory_and_openai_compatible_llm_settings(self):
        calls = []

        class Client:
            base_url = object()
            api_key = "local-key"

        class LLM:
            client = Client()
            model = "gemma4-26b-a4b"
            max_tokens = 2048
            request_timeout_seconds = 42.0
            dspy_temperature = 0.2
            dspy_cache = True

        runtime = DspyRuntime(
            signatures=H3SignatureBundle(object, object, object, object),
            lm_factory=lambda *args, **kwargs: calls.append((args, kwargs)) or "lm",
            predict_factory=lambda signature: signature,
            context_factory=lambda **kwargs: nullcontext(kwargs),
        )

        self.assertEqual("lm", runtime.make_lm(LLM()))
        self.assertEqual(("openai/gemma4-26b-a4b",), calls[0][0])
        self.assertEqual(
            {
                "api_base": str(Client.base_url),
                "api_key": "local-key",
                "temperature": 0.2,
                "max_tokens": 2048,
                "timeout": 42.0,
                "cache": True,
            },
            calls[0][1],
        )

    def test_generator_accepts_fake_runtime_and_loads_h3_guides_without_live_endpoint(self):
        class FakePredict:
            def __init__(self, signature):
                self.signature = signature
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                if self.signature == "Plan":
                    return type("Prediction", (), {
                        "plan": PromptPlan(
                            creative_intent="intent",
                            overall_soundscape="quiet",
                            music_intent=MusicIntent.NONE,
                        )
                    })()
                return type("Prediction", (), {
                    "result": BaseVideoPrompt(
                        integrated_multimodal_description="generated",
                        overall_soundscape="quiet",
                    )
                })()

        predictions = []
        runtime = DspyRuntime(
            signatures=H3SignatureBundle("Analyze", "Plan", "Base", "Reference"),
            lm_factory=lambda *args, **kwargs: "lm",
            predict_factory=lambda signature: predictions.append(FakePredict(signature)) or predictions[-1],
            context_factory=lambda **kwargs: nullcontext(),
        )

        class LLM:
            client = None
            model = "fake"
            max_tokens = 128

        generator = VideoPromptGenerator(
            base_guide_path="minimax-h3-base.md",
            reference_guide_path="minimax-h3-references.md",
            llm=LLM(),
            dspy_runtime=runtime,
        )

        result = generator({
            "mode": "t2v",
            "user_prompt": "A dancer moves.",
            "references": [{"kind": "picture", "source": "actor.png", "description": "actor"}],
            "relay_segments": [{"shot": 1, "start_seconds": 0.0, "end_seconds": 1.0}],
        })

        self.assertEqual("generated", result.prompt.integrated_multimodal_description)
        plan_call = next(predict.calls[0] for predict in predictions if predict.signature == "Plan")
        base_call = next(predict.calls[0] for predict in predictions if predict.signature == "Base")
        self.assertIsInstance(plan_call["references"][0], ResolvedReference)
        self.assertEqual([{"shot": 1, "start_seconds": 0.0, "end_seconds": 1.0}], plan_call["relay_segments"])
        self.assertIsInstance(base_call["plan"], ResolvedPromptPlan)
        self.assertIsInstance(base_call["references"][0], ResolvedReference)
        self.assertNotIn("plan_json", base_call)
        self.assertNotIn("references_json", base_call)
        self.assertNotIn("relay_segments_json", base_call)
        self.assertIn("integrated_multimodal_description", base_call["guide"])


if __name__ == "__main__":
    unittest.main()
