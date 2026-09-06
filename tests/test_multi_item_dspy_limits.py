import unittest
from contextlib import nullcontext
from typing import Any

from feverslop.prompting.general_modules import GeneralPromptModules
from feverslop.prompting.general_signatures import build_general_signature_bundle
from feverslop.prompting.msr_modules import MSRPromptModules


class MultiItemDspyLimitTests(unittest.TestCase):
    def _runtime(self, calls, result):
        class Predictor:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                return result

        class Runtime:
            def make_lm(self, llm):
                return "lm"

            context = staticmethod(lambda **kwargs: nullcontext())
            predict = staticmethod(lambda signature: Predictor())

        return Runtime()

    def test_lyric_alignment_scales_limit_by_segment_count(self):
        calls = []

        class LLM:
            model = "fake-model"
            client = object()

        modules = GeneralPromptModules(
            LLM(),
            dspy_runtime=self._runtime(calls, {"result": {"segments": {}}}),
        )
        modules.lyric_alignment({"WHISPER_SEGMENTS": [{"key": str(i)} for i in range(13)]})

        self.assertEqual(15360, calls[0]["config"]["max_tokens"])

    def test_i2v_signature_transports_optional_performers_without_pydantic_validation(self):
        signature = build_general_signature_bundle()["i2v_prompt"]

        self.assertEqual(dict[str, Any], signature.fields["result"].annotation)

    def test_i2v_prompt_discards_invalid_optional_vocal_performer_metadata(self):
        calls = []

        class LLM:
            model = "fake-model"
            client = object()

        modules = GeneralPromptModules(
            LLM(),
            dspy_runtime=self._runtime(calls, {
                "result": {
                    "prompt": "Mara sings into the rain.",
                    "vocal_performers": [
                        {"subject_id": "mara", "speaker_id": "s1"},
                        {"subject_id": "jon", "speaker_id": "lead vocalist"},
                    ],
                },
            }),
        )

        result = modules.i2v_prompt({}, guide="test guide")

        self.assertEqual("Mara sings into the rain.", result.prompt)
        self.assertEqual(
            [{"subject_id": "mara", "speaker_id": "S1"}],
            [performer.model_dump() for performer in result.vocal_performers],
        )

    def test_msr_segments_scales_limit_by_relay_count(self):
        calls = []

        class LLM:
            model = "fake-model"
            client = object()

        modules = MSRPromptModules(
            LLM(),
            dspy_runtime=self._runtime(calls, {"result": {"references": [], "relays": []}}),
        )
        modules.segments({"relay_segments": [{"index": i} for i in range(4)]})

        self.assertEqual(10240, calls[0]["config"]["max_tokens"])


if __name__ == "__main__":
    unittest.main()
