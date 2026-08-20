import os
import unittest
from unittest.mock import MagicMock, patch
from openai import APIConnectionError, RateLimitError
import httpx
from pathlib import Path
import tempfile

from PIL import Image

from feverslop.adapters.llm_client import LocalOpenAIClient
from feverslop.adapters.api_observability import APIMetrics
from feverslop.errors import FeverSlopLMLError

_saved_allowed_api_hosts = None


def setUpModule():
    # validate_api_url falls back to FEVERSLOP_ALLOWED_API_HOSTS when no
    # allowlist is passed explicitly; keep the ambient value out of these tests.
    global _saved_allowed_api_hosts
    _saved_allowed_api_hosts = os.environ.pop("FEVERSLOP_ALLOWED_API_HOSTS", None)


def tearDownModule():
    global _saved_allowed_api_hosts
    if _saved_allowed_api_hosts is not None:
        os.environ["FEVERSLOP_ALLOWED_API_HOSTS"] = _saved_allowed_api_hosts


class LLMModelCapabilityTests(unittest.TestCase):
    def test_model_metadata_reports_vision_from_modalities(self):
        from feverslop.adapters.llm_client import model_supports_vision

        self.assertTrue(model_supports_vision({"modalities": ["text", "vision"]}))
        self.assertFalse(model_supports_vision({"modalities": ["text"]}))

    def test_missing_model_capability_is_not_assumed_to_be_vision(self):
        from feverslop.adapters.llm_client import model_supports_vision

        self.assertFalse(model_supports_vision({}))

    @patch("feverslop.adapters.llm_client.OpenAI")
    def test_client_reads_vision_capability_from_model_endpoint(self, mock_openai):
        client = MagicMock()
        mock_openai.return_value = client
        client.models.retrieve.return_value = {"id": "vision-model", "modalities": ["text", "vision"]}

        llm = LocalOpenAIClient(api_key="test-key", model="vision-model")

        self.assertTrue(llm.model_supports_vision())
        client.models.retrieve.assert_called_once_with("vision-model")


class LLMClientRetryTests(unittest.TestCase):
    @patch("feverslop.adapters.llm_client.OpenAI")
    def test_health_check_probes_model_endpoint(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.models.list.return_value = object()

        metrics = APIMetrics()
        client = LocalOpenAIClient(api_key="test-key", metrics=metrics)

        self.assertTrue(client.health_check())
        mock_client.models.list.assert_called_once()
        self.assertEqual(1, metrics.snapshot()["llm", "health_check"].successes)

    @patch("feverslop.adapters.llm_client.OpenAI")
    @patch("feverslop.adapters.llm_client.time.sleep", return_value=None)
    def test_records_logical_request_with_explicit_retry_count(self, mock_sleep, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_client.chat.completions.create.side_effect = [
            APIConnectionError(message="connect failed", request=httpx.Request("GET", "http://localhost")),
            APIConnectionError(message="connect failed", request=httpx.Request("GET", "http://localhost")),
            response,
        ]
        metrics = APIMetrics()

        client = LocalOpenAIClient(
            api_key="test-key", max_retries=3, retry_base_delay=0.01, metrics=metrics
        )
        self.assertEqual("ok", client.complete_prompt("system", "prompt"))

        stats = metrics.snapshot()["llm", "chat_completions"]
        self.assertEqual(1, stats.calls)
        self.assertEqual(2, stats.retry_attempts)

    @patch("feverslop.adapters.llm_client.OpenAI")
    @patch("feverslop.adapters.llm_client.time.sleep", return_value=None)
    def test_exhausted_retries_count_only_retries_after_initial_attempt(self, mock_sleep, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = APIConnectionError(
            message="connect failed", request=httpx.Request("GET", "http://localhost")
        )
        metrics = APIMetrics()
        client = LocalOpenAIClient(api_key="test-key", max_retries=3, metrics=metrics)

        with self.assertRaises(FeverSlopLMLError):
            client.complete_prompt("system", "prompt")

        stats = metrics.snapshot()["llm", "chat_completions"]
        self.assertEqual(1, stats.calls)
        self.assertEqual(2, stats.retry_attempts)

    @patch("feverslop.adapters.llm_client.OpenAI")
    def test_captures_usage_finish_reason_and_reasoning_tokens(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="ok"), finish_reason="stop")]
        response.model = "gemma4"
        response.usage = MagicMock(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            completion_tokens_details=MagicMock(reasoning_tokens=12),
        )
        mock_client.chat.completions.create.return_value = response

        client = LocalOpenAIClient(api_key="test-key")
        self.assertEqual("ok", client.complete_prompt(system_prompt="system", prompt="test"))

        telemetry = client.last_response_telemetry
        self.assertEqual((10, 20, 12, "stop"), (
            telemetry.prompt_tokens, telemetry.completion_tokens,
            telemetry.reasoning_tokens, telemetry.finish_reason,
        ))
    @patch("feverslop.adapters.llm_client.OpenAI")
    def test_rejects_private_non_loopback_endpoint(self, mock_openai):
        from feverslop.security.url_validation import APIURLValidationError

        with self.assertRaises(APIURLValidationError):
            LocalOpenAIClient(
                base_url="http://10.0.0.8:8080/v1",
                api_key="test-key",
                allow_private_addresses=False,
            )
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

    @patch("feverslop.adapters.llm_client.RequestRateLimiter")
    @patch("feverslop.adapters.llm_client.OpenAI")
    def test_request_rate_limiter_is_applied(self, mock_openai, limiter_class):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_client.chat.completions.create.return_value = response

        client = LocalOpenAIClient(
            api_key="test-key",
            min_request_interval_seconds=0.5,
        )
        client.complete_prompt(system_prompt="system", prompt="test")

        limiter_class.assert_called_once_with(0.5)
        limiter_class.return_value.wait.assert_called_once_with()

    @patch("feverslop.adapters.llm_client.RequestRateLimiter")
    @patch("feverslop.adapters.llm_client.OpenAI")
    def test_request_rate_limiter_covers_health_and_model_requests(self, mock_openai, limiter_class):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.models.retrieve.return_value = {"modalities": ["text"]}

        client = LocalOpenAIClient(api_key="test-key", min_request_interval_seconds=0.5)
        client.health_check()
        client.model_supports_vision()

        self.assertEqual(2, limiter_class.return_value.wait.call_count)

    @patch("feverslop.adapters.llm_client.OpenAI")
    def test_openai_client_does_not_follow_redirects_or_retry_internally(self, mock_openai):
        LocalOpenAIClient(api_key="test-key")

        kwargs = mock_openai.call_args.kwargs
        self.assertFalse(kwargs["http_client"].follow_redirects)
        self.assertEqual(0, kwargs["max_retries"])

    @patch("feverslop.adapters.llm_client.OpenAI")
    def test_keeps_dspy_temperature_for_prompt_runtime(self, mock_openai):
        client = LocalOpenAIClient(api_key="test-key", dspy_temperature=0.25)

        self.assertEqual(0.25, client.dspy_temperature)

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
