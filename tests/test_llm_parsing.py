import unittest

from autoprompter.application.llm_parsing import extract_json_object


class LLMParsingTests(unittest.TestCase):
    def test_extract_json_object_handles_fenced_json(self):
        text = """```json
{"subject": "person", "locations": ["stage"]}
```"""

        self.assertEqual(
            {"subject": "person", "locations": ["stage"]},
            extract_json_object(text),
        )

    def test_extract_json_object_handles_surrounding_text(self):
        text = 'Here is the object: {"segment_001": "A scene."} Thanks.'

        self.assertEqual({"segment_001": "A scene."}, extract_json_object(text))

    def test_extract_json_object_fails_with_clear_message(self):
        with self.assertRaisesRegex(ValueError, "No JSON object found"):
            extract_json_object("no json here")


if __name__ == "__main__":
    unittest.main()
