import json
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
        from feverslop.errors import FeverSlopLMLError
        with self.assertRaisesRegex(FeverSlopLMLError, "No.*JSON object found"):
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

    def test_extract_json_object_handles_braces_inside_string_values(self):
        """Braces inside JSON string values should not confuse depth counter."""
        text = '{"prompt": "A person wearing {jeans"}'
        result = extract_json_object(text)
        self.assertEqual(result, {"prompt": "A person wearing {jeans"})

    def test_extract_json_object_handles_unbalanced_closing_brace_in_string(self):
        """Unbalanced closing brace inside string should not truncate JSON."""
        text = '{"prompt": "value}"}'
        result = extract_json_object(text)
        self.assertEqual(result, {"prompt": "value}"})

    def test_extract_json_object_handles_nested_braces_in_preamble(self):
        """Preamble contains curly braces — should still find the main JSON object."""
        text = 'Some explanation {not json} follows: {"real": "json", "nested": {"a": 1}}. End.'
        result = extract_json_object(text)
        self.assertEqual(result, {"real": "json", "nested": {"a": 1}})

    def test_extract_json_object_fast_path_clean_json(self):
        """Clean JSON without wrapping is parsed directly."""
        text = '{"key": "value", "count": 42}'
        result = extract_json_object(text)
        self.assertEqual(result, {"key": "value", "count": 42})

    def test_extract_json_object_fast_path_json_array_rejected(self):
        """A JSON array is not a dict and should not be returned."""
        from feverslop.errors import FeverSlopLMLError
        with self.assertRaises(FeverSlopLMLError):
            extract_json_object("[1, 2, 3]")

    def test_extract_json_object_deeply_nested(self):
        """Deeply nested JSON objects are handled."""
        nested = {"level1": {"level2": {"level3": {"level4": {"value": "deep"}}}}}
        text = f"Here is the data: {json.dumps(nested)}. Done."
        result = extract_json_object(text)
        self.assertEqual(result, nested)

    def test_extract_json_object_error_includes_raw_response(self):
        """Parse error includes the raw response text for debugging."""
        from feverslop.errors import FeverSlopLMLError
        raw = "This is not JSON at all, really."
        with self.assertRaises(FeverSlopLMLError) as ctx:
            extract_json_object(raw)
        self.assertIn(raw, str(ctx.exception))

    def test_extract_json_object_escapes_in_strings(self):
        """Escaped quotes and backslashes inside strings don't break parsing."""
        text = r'{"message": "She said \"hello\\world\""}'
        result = extract_json_object(text)
        self.assertEqual(result, {"message": 'She said "hello\\world"'})

    def test_extract_json_object_whitespace_only(self):
        """Whitespace-only input raises an error."""
        from feverslop.errors import FeverSlopLMLError
        with self.assertRaises(FeverSlopLMLError):
            extract_json_object("   \n\t  ")

    def test_extract_json_object_fenced_with_surrounding_text(self):
        """Fenced JSON with text before and after is parsed."""
        text = """Sure! Here's the result:
```json
{"id": 1, "nested": {"a": {"b": "c"}}}
```
Let me know if you need more."""
        result = extract_json_object(text)
        self.assertEqual(result, {"id": 1, "nested": {"a": {"b": "c"}}})


if __name__ == "__main__":
    unittest.main()
