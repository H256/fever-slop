import os
import unittest
from unittest.mock import patch

from feverslop.adapters.llm_client import _resolve_api_key


class TestResolveApiKey(unittest.TestCase):
    def test_explicit_key_returns_key(self):
        self.assertEqual(_resolve_api_key("my-key"), "my-key")

    def test_missing_key_raises_value_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                _resolve_api_key(None)
            self.assertIn("LLM_API_KEY", str(ctx.exception))
            self.assertIn("app_config.json", str(ctx.exception))
            self.assertIn(".env", str(ctx.exception))

    def test_empty_key_raises_value_error(self):
        with self.assertRaises(ValueError):
            _resolve_api_key("")

    def test_hardcoded_placeholder_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            _resolve_api_key("not-needed")
        self.assertIn("not-needed", str(ctx.exception))

    def test_env_var_key_used_when_no_explicit_key(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "env-key"}):
            self.assertEqual(_resolve_api_key(None), "env-key")

    def test_explicit_key_takes_precedence_over_env_var(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "env-key"}):
            self.assertEqual(_resolve_api_key("explicit-key"), "explicit-key")

    def test_whitespace_only_env_var_raises_value_error(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "   "}):
            with self.assertRaises(ValueError):
                _resolve_api_key(None)


if __name__ == "__main__":
    unittest.main()
