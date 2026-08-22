import unittest


class LLMBenchmarkTests(unittest.TestCase):
    def test_benchmark_collects_latency_output_size_and_usage(self):
        from feverslop.tools.llm_benchmark import benchmark_prompts

        class Telemetry:
            prompt_tokens = 10
            completion_tokens = 20
            reasoning_tokens = 12
            total_tokens = 30
            finish_reason = "stop"

        class Client:
            last_response_telemetry = Telemetry()

            def complete_prompt(self, *, system_prompt, prompt):
                return "short answer"

        result = benchmark_prompts(Client(), ["one", "two"])

        self.assertEqual(2, result["requests"])
        self.assertEqual(40, result["tokens"]["completion"])
        self.assertEqual(24, result["tokens"]["reasoning"])
        self.assertEqual(4, result["output"]["words"])
        self.assertEqual(2, len(result["samples"]))
        self.assertGreaterEqual(result["latency_ms"]["total"], 0)

    def test_report_carries_model_and_temperature_settings(self):
        from feverslop.tools.llm_benchmark import benchmark_prompts

        class Client:
            def complete_prompt(self, *, system_prompt, prompt):
                return "ok"

        result = benchmark_prompts(Client(), ["one"], model="gemma4-26b-a4b", temperature=0.2)

        self.assertTrue(result["completed"])
        self.assertEqual("gemma4-26b-a4b", result["model"])
        self.assertEqual(0.2, result["temperature"])

    def test_mid_run_failure_emits_partial_report_with_completed_false(self):
        from feverslop.errors import FeverSlopLMLError
        from feverslop.tools.llm_benchmark import benchmark_prompts

        class Client:
            def __init__(self):
                self.calls = 0

            def complete_prompt(self, *, system_prompt, prompt):
                self.calls += 1
                if self.calls == 3:
                    raise FeverSlopLMLError("boom")
                return "ok"

        result = benchmark_prompts(Client(), ["one", "two", "three", "four"])

        self.assertFalse(result["completed"])
        self.assertEqual(2, result["requests"])
        self.assertEqual(2, len(result["samples"]))
        self.assertIn("boom", result["error"])


if __name__ == "__main__":
    unittest.main()
