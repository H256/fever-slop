import unittest
from unittest.mock import MagicMock, patch
from openai import APIConnectionError, RateLimitError
import httpx
from pathlib import Path
import tempfile

from PIL import Image

from feverslop.adapters.llm_client import LocalOpenAIClient
from feverslop.errors import FeverSlopLMLError


class LLMClientRetryTests(unittest.TestCase):
    @patch("feverslop.adapters.llm_client.OpenAI")
    def test_rejects_private_non_loopback_endpoint(self, mock_openai):
        from feverslop.security.url_validation import APIURLValidationError

        with self.assertRaises(APIURLValidationError):
            LocalOpenAIClient(base_url="http://10.0.0.8:8080/v1", api_key="test-key")
        mock_openai.assert_not_called()

    @patch("feverslop.adapters.llm_client.OpenAI")
    def test_uses_request_timeout_from_init(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_client.chat.completions.create.return_value = mock_resp

        client = LocalOpenAIClient(api_key="test-key", request_timeout_seconds=42.0)
        client.complete_prompt(system_prompt="system", prompt="test")

        create_kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(42.0, create_kwargs["timeout"])

    @patch("feverslop.adapters.llm_client.OpenAI")
    def test_complete_prompt_override_timeout(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_client.chat.completions.create.return_value = mock_resp

        client = LocalOpenAIClient(api_key="test-key", request_timeout_seconds=42.0)
        client.complete_prompt(system_prompt="system", prompt="test", timeout=99.0)

        create_kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(99.0, create_kwargs["timeout"])

    @patch("feverslop.adapters.llm_client.OpenAI")
    def test_complete_prompt_with_images_builds_multimodal_user_content(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content=" grounded "))]
        mock_client.chat.completions.create.return_value = response

        with tempfile.TemporaryDirectory() as tmp:
            actor_path = Path(tmp) / "actor.png"
            location_path = Path(tmp) / "location.png"
            Image.new("RGB", (10, 10), "red").save(actor_path)
            Image.new("RGB", (10, 10), "blue").save(location_path)

            client = LocalOpenAIClient(api_key="test-key")
            result = client.complete_prompt_with_images(
                system_prompt="system",
                prompt="describe",
                image_paths=[actor_path, location_path],
            )

        self.assertEqual("grounded", result)
        messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual("system", messages[0]["role"])
        self.assertEqual("text", messages[1]["content"][0]["type"])
        image_parts = [
            part for part in messages[1]["content"] if part["type"] == "image_url"
        ]
        self.assertEqual(2, len(image_parts))
        for part in image_parts:
            self.assertTrue(
                part["image_url"]["url"].startswith("data:image/jpeg;base64,")
            )

    @patch("time.sleep", return_value=None)
    def test_complete_prompt_with_images_retries_retryable_errors(self, mock_sleep):
        mock_client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_client.chat.completions.create.side_effect = [
            APIConnectionError(
                message="connect failed",
                request=httpx.Request("GET", "http://localhost"),
            ),
            response,
        ]

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "image.png"
            Image.new("RGB", (10, 10), "red").save(image_path)
            client = LocalOpenAIClient(api_key="test-key", max_retries=2, retry_base_delay=0.01)
            client.client = mock_client
            result = client.complete_prompt_with_images(
                system_prompt="system",
                prompt="describe",
                image_paths=[image_path],
            )

        self.assertEqual("ok", result)
        self.assertEqual(2, mock_client.chat.completions.create.call_count)
        mock_sleep.assert_called_once()

    @patch("feverslop.adapters.llm_client.OpenAI")
    @patch("time.sleep", return_value=None)
    def test_retries_on_api_connection_error(self, mock_sleep, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        call_count = 0

        def side_effect(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise APIConnectionError(
                    message="connect failed",
                    request=httpx.Request("GET", "http://localhost"),
                )
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content="  ok  "))]
            return mock_resp

        mock_client.chat.completions.create.side_effect = side_effect

        client = LocalOpenAIClient(api_key="test-key", max_retries=3, retry_base_delay=0.01)
        client.client = mock_client
        result = client.complete_prompt(system_prompt="system", prompt="test")
        self.assertEqual(result, "ok")
        self.assertEqual(call_count, 3)

    @patch("feverslop.adapters.llm_client.OpenAI")
    @patch("time.sleep", return_value=None)
    def test_raises_after_max_retries(self, mock_sleep, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_client.chat.completions.create.side_effect = RateLimitError(
            "rate limited", response=MagicMock(), body=None
        )

        client = LocalOpenAIClient(api_key="test-key", max_retries=3, retry_base_delay=0.01)
        client.client = mock_client
        with self.assertRaises(FeverSlopLMLError) as ctx:
            client.complete_prompt(system_prompt="system", prompt="test")
        self.assertIn("3 attempts", str(ctx.exception))

    @patch("time.sleep", return_value=None)
    def test_sleep_called_with_exponential_backoff(self, mock_sleep):
        mock_client = MagicMock()

        call_count = 0

        def side_effect(*a, **kw):
            nonlocal call_count
            call_count += 1
            raise APIConnectionError(
                message="fail",
                request=httpx.Request("GET", "http://localhost"),
            )

        mock_client.chat.completions.create.side_effect = side_effect

        client = LocalOpenAIClient(api_key="test-key", max_retries=3, retry_base_delay=0.5)
        client.client = mock_client
        with self.assertRaises(FeverSlopLMLError):
            client.complete_prompt(system_prompt="system", prompt="test")

        sleep_calls = [c[0][0] for c in mock_sleep.call_args_list]
        self.assertGreaterEqual(sleep_calls[0], 0.5)
        self.assertLess(sleep_calls[0], 1.5)
        self.assertGreaterEqual(sleep_calls[1], 1.0)
        self.assertLess(sleep_calls[1], 3.0)

    @patch("time.sleep", return_value=None)
    def test_retry_delays_include_jitter(self, mock_sleep):
        mock_client = MagicMock()

        call_count = 0

        def side_effect(*a, **kw):
            nonlocal call_count
            call_count += 1
            raise APIConnectionError(
                message="fail",
                request=httpx.Request("GET", "http://localhost"),
            )

        mock_client.chat.completions.create.side_effect = side_effect

        client = LocalOpenAIClient(api_key="test-key", max_retries=5, retry_base_delay=0.1)
        client.client = mock_client
        with self.assertRaises(FeverSlopLMLError):
            client.complete_prompt(system_prompt="system", prompt="test")

        sleep_calls = [c[0][0] for c in mock_sleep.call_args_list]
        self.assertEqual(len(sleep_calls), 4)
        self.assertNotAlmostEqual(sleep_calls[0], sleep_calls[1], places=2)

    @patch("feverslop.adapters.llm_client.OpenAI")
    def test_timeout_passed_to_create(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="result"))]
        mock_client.chat.completions.create.return_value = mock_resp

        client = LocalOpenAIClient(api_key="test-key")
        client.client = mock_client
        client.complete_prompt(system_prompt="system", prompt="test", timeout=60.0)

        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["timeout"], 60.0)

    @patch("feverslop.adapters.llm_client.OpenAI")
    def test_uses_configured_timeout_when_call_does_not_override_it(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="result"))]
        mock_client.chat.completions.create.return_value = mock_resp

        client = LocalOpenAIClient(api_key="test-key", request_timeout_seconds=45.0)
        client.client = mock_client
        client.complete_prompt(system_prompt="system", prompt="test")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(45.0, call_kwargs["timeout"])

    def test_rejects_non_positive_default_timeout(self):
        with self.assertRaisesRegex(ValueError, "request_timeout_seconds"):
            LocalOpenAIClient(api_key="test-key", request_timeout_seconds=0)

    @patch("feverslop.adapters.llm_client.OpenAI")
    def test_no_retry_on_success(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="hello"))]
        mock_client.chat.completions.create.return_value = mock_resp

        client = LocalOpenAIClient(api_key="test-key", max_retries=3, retry_base_delay=0.01)
        client.client = mock_client
        result = client.complete_prompt(system_prompt="system", prompt="hi")
        self.assertEqual(result, "hello")
        self.assertEqual(mock_client.chat.completions.create.call_count, 1)


if __name__ == "__main__":
    unittest.main()
