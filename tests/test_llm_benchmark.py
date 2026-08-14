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


if __name__ == "__main__":
    unittest.main()
