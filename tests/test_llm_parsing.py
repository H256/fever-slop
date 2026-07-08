import unittest

from feverslop.application.llm_parsing import extract_json_object


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
        with self.assertRaisesRegex(ValueError, "No.*JSON object found"):
            extract_json_object("no json here")

    def test_extract_json_object_handles_trailing_text_after_fence(self):
        """LLM returns JSON in a fence followed by commentary."""
        text = """```json
    {"title": "Test", "beats": ["beat one"]}
    Hope this helps!"""
        result = extract_json_object(text)
        self.assertEqual(result, {"title": "Test", "beats": ["beat one"]})

    def test_extract_json_object_handles_preamble_before_fence(self):
        """LLM returns commentary before the fenced JSON."""
        text = """Here is the screenplay structure:
    {"title": "Test"}
    ```"""
        result = extract_json_object(text)
        self.assertEqual(result, {"title": "Test"})

    def test_extract_json_object_handles_uppercase_json_tag(self):
        """LLM uses uppercase JSON language tag in code fence."""
        text = """```JSON
    {"key": "value"}
    ```"""
        result = extract_json_object(text)
        self.assertEqual(result, {"key": "value"})

    def test_extract_json_object_handles_nested_braces_in_preamble(self):
        """Preamble contains curly braces — should still find the main JSON object."""
        text = 'Some explanation {not json} follows: {"real": "json", "nested": {"a": 1}}. End.'
        result = extract_json_object(text)
        self.assertEqual(result, {"real": "json", "nested": {"a": 1}})

if __name__ == "__main__":
    unittest.main()
