from __future__ import annotations

import unittest

from feverslop.prompting.general_signatures import parse_prompt_result


class PromptResultParsingTests(unittest.TestCase):
    def test_unwraps_dspy_result_envelope(self):
        result = parse_prompt_result({
            "result": {
                "prompt": "A woman crosses the glass chamber.",
                "vocal_performers": [],
            },
        })

        self.assertEqual("A woman crosses the glass chamber.", result.prompt)
        self.assertEqual([], result.vocal_performers)


if __name__ == "__main__":
    unittest.main()
