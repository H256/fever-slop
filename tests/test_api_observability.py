import unittest
from unittest.mock import MagicMock, patch

from feverslop.adapters.api_observability import (
    APIMetrics,
    RequestRateLimiter,
    api_observability_context,
    record_api_call,
    redact_secrets,
    require_json_object,
)


class APIMetricsTests(unittest.TestCase):
    def test_require_json_object_rejects_scalar_external_payloads(self):
        with self.assertRaises(ValueError):
            require_json_object([], context="test")

    def test_rate_limiter_rejects_negative_interval_and_is_disabled_by_default(self):
        with self.assertRaises(ValueError):
            RequestRateLimiter(-1)
        RequestRateLimiter().wait()

    def test_redact_secrets_removes_query_and_header_style_credentials(self):
        value = "https://llm.example/v1?api_key=secret123&model=x Authorization: Bearer abc123"
        redacted = redact_secrets(value)
        self.assertNotIn("secret123", redacted)
        self.assertNotIn("abc123", redacted)
        self.assertIn("api_key=[REDACTED]", redacted)
        self.assertIn("[REDACTED]", redacted)

    def test_records_success_and_failure_by_service_and_operation(self):
        metrics = APIMetrics()
        metrics.record("comfyui", "queue_prompt", 12.5, success=True)
        metrics.record("comfyui", "queue_prompt", 7.5, success=False)

        stats = metrics.snapshot()[("comfyui", "queue_prompt")]
        self.assertEqual(2, stats.calls)
        self.assertEqual(1, stats.successes)
        self.assertEqual(1, stats.failures)
        self.assertEqual(20.0, stats.total_duration_ms)

    def test_records_usage_and_cost(self):
        metrics = APIMetrics()
        metrics.record("llm", "chat", 1, success=True, usage_units=100, estimated_cost=0.02)
        stats = metrics.snapshot()[("llm", "chat")]
        self.assertEqual(100, stats.usage_units)
        self.assertEqual(0.02, stats.estimated_cost)

    def test_records_token_breakdown_without_prompt_content(self):
        metrics = APIMetrics()
        metrics.record(
            "llm", "chat", 1, success=True,
            prompt_tokens=10, completion_tokens=20, reasoning_tokens=12,
        )

        stats = metrics.snapshot()["llm", "chat"]
        self.assertEqual((10, 20, 12), (stats.prompt_tokens, stats.completion_tokens, stats.reasoning_tokens))
        entry = metrics.export_snapshot()["entries"][0]
        self.assertEqual(10, entry["prompt_tokens"])
        self.assertEqual(12, entry["reasoning_tokens"])

    def test_export_snapshot_has_stable_json_schema(self):
        metrics = APIMetrics()
        metrics.record("llm", "chat", 1, success=True)
        snapshot = metrics.export_snapshot()
        self.assertEqual(1, snapshot["version"])
        self.assertEqual("llm", snapshot["entries"][0]["service"])
        self.assertIn('"entries"', metrics.export_json())

    def test_records_correlation_context_and_generates_request_id(self):
        metrics = APIMetrics()

        with api_observability_context(job_id="job-1", project_id="project-1", scene_id="scene-2"):
            record_api_call(metrics, None, "llm", "chat", 0.0, success=True)

        entry = metrics.export_snapshot()["entries"][0]
        self.assertTrue(entry["correlation_id"])
        self.assertEqual("job-1", entry["job_id"])
        self.assertEqual("project-1", entry["project_id"])
        self.assertEqual("scene-2", entry["scene_id"])

    def test_exposes_percentiles_retry_count_and_time_window(self):
        metrics = APIMetrics()
        for duration in (10, 20, 30, 40, 50):
            metrics.record("llm", "chat", duration, success=True, retry_attempts=1, timestamp=4.0)

        stats = metrics.snapshot()["llm", "chat"]
        self.assertEqual(5, stats.retry_attempts)
        self.assertEqual(30.0, stats.p50_duration_ms)
        self.assertEqual(50.0, stats.p95_duration_ms)
        self.assertEqual(50.0, stats.p99_duration_ms)

        metrics.record("llm", "chat", 100, success=False, timestamp=1.0)
        windowed = metrics.snapshot(window_seconds=2, now=5.0)["llm", "chat"]
        self.assertEqual(5, windowed.calls)
        self.assertIn('"entries"', metrics.export_json(window_seconds=2, now=5.0))

    def test_retains_only_the_configured_number_of_recent_samples(self):
        metrics = APIMetrics(max_samples_per_operation=2)
        metrics.record("llm", "chat", 10, success=True, timestamp=1.0)
        metrics.record("llm", "chat", 20, success=True, timestamp=2.0)
        metrics.record("llm", "chat", 30, success=True, timestamp=3.0)

        stats = metrics.snapshot()["llm", "chat"]
        self.assertEqual(3, stats.calls)
        self.assertEqual(20.0, stats.p50_duration_ms)

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
        self.assertIn(f"correlation_id={metrics.export_snapshot()['entries'][0]['correlation_id']}", logs.output[0])

    @patch("requests.Session")
    def test_comfyui_health_check_uses_read_only_probe(self, session_class):
        from feverslop.adapters.comfyui_client import ComfyUIClient

        session = MagicMock()
        session.get.return_value.ok = True
        session_class.return_value = session

        self.assertTrue(ComfyUIClient().health_check())
        session.get.assert_called_once()
        self.assertIn("system_stats", session.get.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
