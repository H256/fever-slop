import unittest
from unittest.mock import MagicMock, patch
from openai import APIConnectionError, RateLimitError
import httpx

from feverslop.adapters.llm_client import LocalOpenAIClient
from feverslop.errors import FeverSlopLMLError


class LLMClientRetryTests(unittest.TestCase):
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

        client = LocalOpenAIClient(max_retries=3, retry_base_delay=0.01)
        client.client = mock_client
        result = client.complete_prompt("test")
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

        client = LocalOpenAIClient(max_retries=3, retry_base_delay=0.01)
        client.client = mock_client
        with self.assertRaises(FeverSlopLMLError) as ctx:
            client.complete_prompt("test")
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

        client = LocalOpenAIClient(max_retries=3, retry_base_delay=0.5)
        client.client = mock_client
        with self.assertRaises(FeverSlopLMLError):
            client.complete_prompt("test")

        sleep_calls = [c[0][0] for c in mock_sleep.call_args_list]
        self.assertAlmostEqual(sleep_calls[0], 0.5, places=1)
        self.assertAlmostEqual(sleep_calls[1], 1.0, places=1)

    @patch("feverslop.adapters.llm_client.OpenAI")
    def test_no_retry_on_success(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="hello"))]
        mock_client.chat.completions.create.return_value = mock_resp

        client = LocalOpenAIClient(max_retries=3, retry_base_delay=0.01)
        client.client = mock_client
        result = client.complete_prompt("hi")
        self.assertEqual(result, "hello")
        self.assertEqual(mock_client.chat.completions.create.call_count, 1)


if __name__ == "__main__":
    unittest.main()
