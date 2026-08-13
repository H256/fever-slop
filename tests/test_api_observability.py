import unittest
from unittest.mock import MagicMock, patch

from feverslop.adapters.api_observability import APIMetrics


class APIMetricsTests(unittest.TestCase):
    def test_records_success_and_failure_by_service_and_operation(self):
        metrics = APIMetrics()
        metrics.record("comfyui", "queue_prompt", 12.5, success=True)
        metrics.record("comfyui", "queue_prompt", 7.5, success=False)

        stats = metrics.snapshot()[("comfyui", "queue_prompt")]
        self.assertEqual(2, stats.calls)
        self.assertEqual(1, stats.successes)
        self.assertEqual(1, stats.failures)
        self.assertEqual(20.0, stats.total_duration_ms)

    @patch("requests.Session")
    def test_comfyui_client_records_http_call_and_structured_log(self, session_class):
        from feverslop.adapters.comfyui_client import ComfyUIClient

        session = MagicMock()
        session.post.return_value.ok = True
        session.post.return_value.json.return_value = {"prompt_id": "p1"}
        session_class.return_value = session
        metrics = APIMetrics()

        with self.assertLogs("feverslop.adapters.comfyui_client", level="INFO") as logs:
            ComfyUIClient(metrics=metrics).queue_prompt({})

        stats = metrics.snapshot()[("comfyui", "queue_prompt")]
        self.assertEqual((1, 1, 0), (stats.calls, stats.successes, stats.failures))
        self.assertIn("api_call service=comfyui operation=queue_prompt", logs.output[0])


if __name__ == "__main__":
    unittest.main()
