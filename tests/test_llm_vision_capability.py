import unittest
from unittest.mock import MagicMock, patch

from feverslop.adapters.llm_client import LocalOpenAIClient

VISION_MODEL = {"modalities": ["text", "vision"]}
TEXT_MODEL = {"modalities": ["text"]}


class LLMVisionCapabilityProbeTests(unittest.TestCase):
    @patch("feverslop.adapters.llm_client.OpenAI")
    def test_probe_uses_bounded_timeout(self, mock_openai):
        with self.subTest(msg="default_timeout"):
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.models.retrieve.return_value = VISION_MODEL
            client = LocalOpenAIClient(api_key="test-key", model="probe-model")

            client.model_supports_vision()

            mock_client.models.retrieve.assert_called_once_with("probe-model", timeout=10.0)

        with self.subTest(msg="short_timeout"):
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.models.retrieve.return_value = VISION_MODEL
            client = LocalOpenAIClient(
                api_key="test-key", model="probe-model", request_timeout_seconds=4.0,
            )

            client.model_supports_vision()

            mock_client.models.retrieve.assert_called_once_with("probe-model", timeout=4.0)

    @patch("feverslop.adapters.llm_client.RequestRateLimiter")
    @patch("feverslop.adapters.llm_client.OpenAI")
    def test_capability_is_cached_after_successful_probe(self, mock_openai, limiter_class):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        cases = (
            ("vision model", VISION_MODEL, True),
            ("text model", TEXT_MODEL, False),
        )
        for label, model_info, expected in cases:
            with self.subTest(label):
                mock_client.models.retrieve.reset_mock()
                limiter_class.return_value.wait.reset_mock()
                mock_client.models.retrieve.return_value = model_info
                client = LocalOpenAIClient(api_key="test-key", model="probe-model")

                self.assertEqual(expected, client.model_supports_vision())
                self.assertEqual(expected, client.model_supports_vision())

                self.assertEqual(1, mock_client.models.retrieve.call_count)
                self.assertEqual(1, limiter_class.return_value.wait.call_count)

    @patch("feverslop.adapters.llm_client.OpenAI")
    def test_failed_probe_propagates_and_is_not_cached(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.models.retrieve.side_effect = RuntimeError("probe outage")

        client = LocalOpenAIClient(api_key="test-key", model="probe-model")

        with self.assertRaises(RuntimeError):
            client.model_supports_vision()
        self.assertIsNone(client._vision_capability)

        mock_client.models.retrieve.side_effect = None
        mock_client.models.retrieve.return_value = VISION_MODEL
        self.assertTrue(client.model_supports_vision())
        self.assertEqual(2, mock_client.models.retrieve.call_count)
        self.assertTrue(client.model_supports_vision())
        self.assertEqual(2, mock_client.models.retrieve.call_count)


if __name__ == "__main__":
    unittest.main()
