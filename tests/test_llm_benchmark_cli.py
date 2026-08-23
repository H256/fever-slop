from contextlib import redirect_stderr, redirect_stdout
import json
import os
import tempfile
from io import StringIO
import unittest
from unittest import mock

from feverslop.errors import FeverSlopLMLError
from feverslop.tools import llm_benchmark
from feverslop.tools.llm_benchmark import main


class LLMBenchmarkCliTests(unittest.TestCase):
    def test_missing_llm_api_key_exits_with_named_error(self):
        stdout = StringIO()
        stderr = StringIO()
        with mock.patch.dict(os.environ, {"LLM_API_KEY": ""}):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                with self.assertRaises(SystemExit):
                    main(["--prompt-file", "missing.txt"])

        self.assertEqual("", stdout.getvalue())
        self.assertIn("LLM_API_KEY environment variable is required", stderr.getvalue())

    def test_api_key_flag_is_rejected(self):
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                main(["--prompt-file", "missing.txt", "--api-key", "x"])

        self.assertEqual("", stdout.getvalue())
        self.assertIn("unrecognized", stderr.getvalue())

    def test_temperature_flag_reaches_client_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_file = os.path.join(tmp, "prompts.txt")
            with open(prompt_file, "w", encoding="utf-8") as handle:
                handle.write("hello\n")

            created = {}

            class FakeClient:
                def __init__(self, **kwargs):
                    created.update(kwargs)

                def complete_prompt(self, *, system_prompt, prompt):
                    return "ok"

            stdout = StringIO()
            stderr = StringIO()
            with mock.patch.dict(os.environ, {"LLM_API_KEY": "test-key"}):
                with mock.patch.object(llm_benchmark, "OpenAICompatibleLLMClient", FakeClient):
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        exit_code = main(["--prompt-file", prompt_file, "--temperature", "0.2", "--model", "mymodel"])

            self.assertEqual(0, exit_code)
            self.assertEqual(0.2, created["temperature"])
            self.assertNotIn("dspy_temperature", created)
            report = json.loads(stdout.getvalue())
            self.assertEqual(0.2, report["temperature"])
            self.assertEqual("mymodel", report["model"])

    def test_dspy_temperature_flag_is_rejected(self):
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                main(["--prompt-file", "missing.txt", "--dspy-temperature", "0.1"])

        self.assertEqual("", stdout.getvalue())
        self.assertIn("unrecognized", stderr.getvalue())

    def test_main_returns_nonzero_and_prints_partial_report_on_llm_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_file = os.path.join(tmp, "prompts.txt")
            with open(prompt_file, "w", encoding="utf-8") as handle:
                handle.write("one\ntwo\nthree\n")

            class FailingClient:
                def __init__(self, **kwargs):
                    self.calls = 0

                def complete_prompt(self, *, system_prompt, prompt):
                    self.calls += 1
                    if self.calls == 2:
                        raise FeverSlopLMLError("endpoint down")
                    return "ok"

            stdout = StringIO()
            stderr = StringIO()
            with mock.patch.dict(os.environ, {"LLM_API_KEY": "test-key"}):
                with mock.patch.object(llm_benchmark, "OpenAICompatibleLLMClient", FailingClient):
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        exit_code = main(["--prompt-file", prompt_file])

            self.assertEqual(1, exit_code)
            report = json.loads(stdout.getvalue())
            self.assertFalse(report["completed"])
            self.assertEqual(1, len(report["samples"]))
            self.assertIn("endpoint down", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
