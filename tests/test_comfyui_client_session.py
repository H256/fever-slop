from __future__ import annotations

import unittest
from threading import Event
from unittest.mock import MagicMock, patch

from feverslop.adapters.comfyui_client import ComfyUIClient


class ComfyUIClientSessionTest(unittest.TestCase):
    def test_client_exposes_input_file_preflight(self):
        self.assertTrue(hasattr(ComfyUIClient, "input_file_exists"))

    def test_input_file_preflight_normalizes_server_path(self):
        client = ComfyUIClient(base_url="http://test:8188")
        response = MagicMock(ok=True, status_code=200)
        session = MagicMock()
        session.get.return_value = response
        client._session = session

        exists = client.input_file_exists(r"feverslop\audio\song.wav")

        self.assertTrue(exists)
        session.get.assert_called_once_with(
            "http://test:8188/view",
            params={"filename": "song.wav", "subfolder": "feverslop/audio", "type": "input"},
            timeout=60,
            stream=True,
            allow_redirects=False,
        )
        response.close.assert_called_once_with()

    def test_input_file_preflight_returns_false_for_404(self):
        client = ComfyUIClient(base_url="http://test:8188")
        response = MagicMock(ok=False, status_code=404)
        session = MagicMock()
        session.get.return_value = response
        client._session = session

        self.assertFalse(client.input_file_exists("missing.png"))
        response.close.assert_called_once_with()

    def test_session_is_reused_across_calls(self):
        """Verify that a single session is created and reused."""
        client = ComfyUIClient(base_url="http://test:8188")
        self.assertIsNone(client._session)

        with patch("requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session

            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.json.return_value = {"prompt_id": "abc123"}
            mock_session.post.return_value = mock_response

            client.queue_prompt({"test": 1})
            client.queue_prompt({"test": 2})

            mock_session_cls.assert_called_once()
            self.assertEqual(mock_session.post.call_count, 2)
            self.assertEqual(2, mock_session.mount.call_count)
            mock_session.mount.assert_any_call("http://", unittest.mock.ANY)
            mock_session.mount.assert_any_call("https://", unittest.mock.ANY)

    def test_wait_for_completion_honors_cancellation_event(self):
        client = ComfyUIClient(base_url="http://test:8188")
        cancel_event = Event()
        cancel_event.set()
        with self.assertRaisesRegex(InterruptedError, "cancelled"):
            client.wait_for_completion("prompt", cancel_event=cancel_event)

    def test_session_created_on_first_request(self):
        """Session should be created lazily on first request."""
        client = ComfyUIClient(base_url="http://test:8188")
        self.assertIsNone(client._session)

        with patch("requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session

            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.json.return_value = {}
            mock_session.get.return_value = mock_response

            client.get_object_info()
            mock_session_cls.assert_called_once()
            self.assertIsNotNone(client._session)

    def test_close_disposes_session(self):
        """close() should close and clear the session."""
        client = ComfyUIClient(base_url="http://test:8188")

        with patch("requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session

            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.json.return_value = {"prompt_id": "abc"}
            mock_session.post.return_value = mock_response

            client.queue_prompt({"test": 1})
            self.assertIsNotNone(client._session)

            client.close()
            mock_session.close.assert_called_once()
            self.assertIsNone(client._session)

    def test_close_is_idempotent(self):
        """Calling close() multiple times should be safe."""
        client = ComfyUIClient(base_url="http://test:8188")
        client.close()
        client.close()
        self.assertIsNone(client._session)

    def test_context_manager_closes_session(self):
        """Using client as context manager should close session on exit."""
        with patch("requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session

            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.json.return_value = {"prompt_id": "abc"}
            mock_session.post.return_value = mock_response

            with ComfyUIClient(base_url="http://test:8188") as client:
                client.queue_prompt({"test": 1})
                self.assertIsNotNone(client._session)

            mock_session.close.assert_called_once()
            self.assertIsNone(client._session)

    def test_session_reused_for_get_and_post(self):
        """Same session should be used for both GET and POST requests."""
        client = ComfyUIClient(base_url="http://test:8188")

        with patch("requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session

            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.json.return_value = {"prompt_id": "abc"}
            mock_session.post.return_value = mock_response
            mock_session.get.return_value = mock_response

            client.queue_prompt({"test": 1})
            client.get_object_info()

            mock_session_cls.assert_called_once()
            mock_session.post.assert_called_once()
            mock_session.get.assert_called_once()

    def test_new_session_after_close(self):
        """After close(), a new session should be created on next request."""
        client = ComfyUIClient(base_url="http://test:8188")

        with patch("requests.Session") as mock_session_cls:
            mock_session1 = MagicMock()
            mock_session2 = MagicMock()
            mock_session_cls.side_effect = [mock_session1, mock_session2]

            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.json.return_value = {"prompt_id": "abc"}
            mock_session1.post.return_value = mock_response
            mock_session2.post.return_value = mock_response

            client.queue_prompt({"test": 1})
            client.close()
            client.queue_prompt({"test": 2})

            self.assertEqual(mock_session_cls.call_count, 2)
            mock_session1.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
