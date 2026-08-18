import unittest
from contextlib import nullcontext

from feverslop.prompting.general_modules import GeneralPromptModules
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

        self.assertEqual(4352, calls[0]["config"]["max_tokens"])

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

        self.assertEqual(3072, calls[0]["config"]["max_tokens"])


if __name__ == "__main__":
    unittest.main()
