import unittest

from feverslop.application.prompt_generation import PromptGenerationService
from feverslop.prompting.dspy_h3_models import (
    BaseVideoPrompt,
    GeneratedVideoPrompt,
    MusicIntent,
    PromptMode,
    ResolvedPromptPlan,
)
from feverslop.prompting.model_types import resolve_model_type


def generated_prompt(mode=PromptMode.T2V):
    return GeneratedVideoPrompt(
        mode=mode,
        prompt=BaseVideoPrompt(
            integrated_multimodal_description="generated",
            overall_soundscape="quiet",
        ),
        plan=ResolvedPromptPlan(
            creative_intent="test",
            overall_soundscape="quiet",
            music_intent=MusicIntent.NONE,
        ),
        references=[],
    )


class RecordingGenerator:
    def __init__(self, result):
        self.requests = []
        self.result = result

    def __call__(self, request):
        self.requests.append(request)
        return self.result


class PromptGenerationServiceTests(unittest.TestCase):
    def test_builds_request_for_resolved_model_type_and_returns_generator_result(self):
        generator = RecordingGenerator(generated_prompt())
        service = PromptGenerationService(generator)

        result = service.generate(
            resolve_model_type("minimax-h3-t2v"),
            "A dancer crosses a neon-lit street.",
            references=[{"kind": "picture", "source": "style.png", "role": "style"}],
            notes="Keep the camera low.",
            duration_seconds=6,
            music_intent="none",
            strict_fidelity=False,
        )

        self.assertIs(result, generator.result)
        self.assertEqual(
            {
                "mode": "t2v",
                "user_prompt": "A dancer crosses a neon-lit street.",
                "references": [
                    {"kind": "picture", "source": "style.png", "role": "style", "description": None,
                     "name": None, "use_audio": False},
                ],
                "notes": "Keep the camera low.",
                "duration_seconds": 6.0,
                "music_intent": "none",
                "relay_segments": [],
                "strict_fidelity": False,
            },
            generator.requests[0],
        )

    def test_blank_description_is_rejected_before_generator_call(self):
        generator = RecordingGenerator(generated_prompt())
        service = PromptGenerationService(generator)

        with self.assertRaises(ValueError):
            service.generate("minimax-h3-t2v", "   ")

        self.assertEqual([], generator.requests)

    def test_invalid_mode_reference_combination_is_rejected_before_generator_call(self):
        generator = RecordingGenerator(generated_prompt(PromptMode.I2V))
        service = PromptGenerationService(generator)

        with self.assertRaises(ValueError):
            service.generate("minimax-h3-i2v", "A subject moves.")

        self.assertEqual([], generator.requests)


if __name__ == "__main__":
    unittest.main()
