from contextlib import redirect_stderr, redirect_stdout
import os
from io import StringIO
import unittest
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()
