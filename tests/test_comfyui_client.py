import unittest
from unittest.mock import MagicMock, patch

import requests

from feverslop.errors import FeverSlopWorkflowError


class ComfyUIClientTests(unittest.TestCase):
    @patch("requests.Session")
    def test_optional_auth_header_is_forwarded(self, session_class):
        from feverslop.adapters.comfyui_client import ComfyUIClient

        session = MagicMock()
        session.post.return_value.ok = True
        session.post.return_value.json.return_value = {"prompt_id": "p1"}
        session_class.return_value = session
        ComfyUIClient(api_key="secret").queue_prompt({})
        self.assertEqual("Bearer secret", session.post.call_args.kwargs["headers"]["Authorization"])

    @patch("feverslop.adapters.comfyui_client.RequestRateLimiter")
    @patch("requests.Session")
    def test_request_rate_limiter_is_applied(self, session_class, limiter_class):
        from feverslop.adapters.comfyui_client import ComfyUIClient

        session = MagicMock()
        session.post.return_value.ok = True
        session.post.return_value.json.return_value = {"prompt_id": "p1"}
        session_class.return_value = session

        client = ComfyUIClient(min_request_interval_seconds=0.25)
        client.queue_prompt({})

        limiter_class.assert_called_once_with(0.25)
        limiter_class.return_value.wait.assert_called_once_with()

    def test_rejects_private_non_loopback_endpoint(self):
        from feverslop.adapters.comfyui_client import ComfyUIClient
        from feverslop.security.url_validation import APIURLValidationError

        with self.assertRaises(APIURLValidationError):
            ComfyUIClient(base_url="http://192.168.1.10:8188", allow_private_addresses=False)

    def test_wait_for_completion_raises_comfyui_execution_error_details(self):
        from feverslop.adapters.comfyui_client import ComfyUIClient

        prompt_id = "prompt-oom"
        client = ComfyUIClient(base_url="http://comfy.example")
        client.get_history = MagicMock(return_value={
            prompt_id: {
                "status": {
                    "status_str": "error",
                    "completed": False,
                    "messages": [[
                        "execution_error",
                        {
                            "prompt_id": prompt_id,
                            "node_id": "5012",
                            "node_type": "LTXAddVideoICLoRAGuide",
                            "exception_type": "torch.OutOfMemoryError",
                            "exception_message": "HIP out of memory. Tried to allocate 6.75 GiB.",
                        },
                    ]],
                }
            }
        })

        with self.assertRaisesRegex(
            FeverSlopWorkflowError,
            r"prompt-oom.*node 5012.*LTXAddVideoICLoRAGuide.*torch\.OutOfMemoryError.*6\.75 GiB",
        ):
            client.wait_for_completion(prompt_id, poll_interval=0)

    def test_wait_for_completion_uses_configured_default_timeout(self):
        from feverslop.adapters.comfyui_client import ComfyUIClient

        client = ComfyUIClient(base_url="http://comfy.example", prompt_timeout_seconds=12)

        self.assertEqual(12, client.prompt_timeout_seconds)

    def test_http_requests_use_configured_prompt_timeout(self):
        from feverslop.adapters.comfyui_client import ComfyUIClient

        with patch("requests.Session") as session_class:
            session = MagicMock()
            session.post.return_value.json.return_value = {"prompt_id": "prompt-1"}
            session.get.return_value.json.return_value = {}
            session_class.return_value = session

            client = ComfyUIClient(base_url="http://comfy.example", prompt_timeout_seconds=900)
            client.queue_prompt({})
            client.get_history("prompt-1")
            client.get_object_info()

        session.post.assert_called_once_with(
            "http://comfy.example/prompt",
            json={"prompt": {}, "client_id": client.client_id},
            timeout=900,
            allow_redirects=False,
        )
        session.get.assert_any_call("http://comfy.example/history/prompt-1", timeout=900, allow_redirects=False)
        session.get.assert_any_call("http://comfy.example/object_info", timeout=900, allow_redirects=False)

    def test_http_error_includes_response_body(self):
        from feverslop.adapters.comfyui_client import ComfyUIClient, ComfyUIHTTPError

        response = requests.Response()
        response.status_code = 400
        response.url = "http://comfy.example/prompt"
        response._content = b'{"error":"invalid prompt"}'
        response.headers["Content-Type"] = "application/json"

        client = ComfyUIClient(base_url="http://comfy.example")

        with self.assertRaisesRegex(
            ComfyUIHTTPError,
            r"ComfyUI queue prompt failed with HTTP 400.*invalid prompt",
        ):
            client._raise_for_status(response, "queue prompt")

    def test_redirect_response_is_rejected(self):
        from feverslop.adapters.comfyui_client import ComfyUIClient, ComfyUIHTTPError

        response = requests.Response()
        response.status_code = 302
        response.url = "http://comfy.example/prompt"
        response.headers["Location"] = "http://other.example/prompt"

        with self.assertRaises(ComfyUIHTTPError):
            ComfyUIClient(base_url="http://comfy.example")._raise_for_status(response, "queue prompt")

    def test_free_cache_and_vram_posts_to_comfyui(self):
        from feverslop.adapters.comfyui_client import ComfyUIClient

        with patch("requests.Session") as session_class:
            session = MagicMock()
            session.post.return_value.ok = True
            session_class.return_value = session

            ComfyUIClient(base_url="http://comfy.example/").free_cache_and_vram()

        session.post.assert_called_once_with(
            "http://comfy.example/free",
            json={"unload_models": True, "free_memory": True},
            timeout=30,
            allow_redirects=False,
        )

    def test_free_cache_and_vram_is_best_effort(self):
        from feverslop.adapters.comfyui_client import ComfyUIClient

        with patch("requests.Session") as session_class:
            session_class.return_value.post.side_effect = requests.RequestException("offline")

            with self.assertLogs("feverslop.adapters.comfyui_client", level="DEBUG") as logs:
                ComfyUIClient().free_cache_and_vram()

        self.assertIn("ComfyUI cache/VRAM release failed: offline", logs.output[0])


if __name__ == "__main__":
    unittest.main()
