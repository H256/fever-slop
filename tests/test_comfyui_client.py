import unittest

import requests


class ComfyUIClientTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
