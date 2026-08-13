from __future__ import annotations

import logging
import os
from pathlib import Path
import json
import time
import uuid
import requests

from feverslop.adapters.api_observability import APIMetrics, default_api_metrics, record_api_call, redact_secrets, require_json_object
from feverslop.errors import FeverSlopWorkflowError
from feverslop.security.url_validation import validate_api_url

logger = logging.getLogger(__name__)


class ComfyUIHTTPError(FeverSlopWorkflowError):
    pass


class ComfyUIExecutionError(FeverSlopWorkflowError):
    pass


def json_dumps_compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class ComfyUIClient:
    """
    Minimal ComfyUI HTTP API client.

    Responsibilities:
    - queue workflow API JSON
    - poll history until done
    - upload input assets
    - download generated output assets
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8188",
        client_id: str | None = None,
        prompt_timeout_seconds: float = 1800.0,
        metrics: APIMetrics | None = None,
        api_key: str | None = None,
        auth_header: str = "Authorization",
    ):
        self.base_url = validate_api_url(base_url).rstrip("/")
        self.client_id = client_id or str(uuid.uuid4())
        self.prompt_timeout_seconds = float(prompt_timeout_seconds)
        self.metrics = metrics or default_api_metrics
        token = api_key or os.environ.get("COMFYUI_API_KEY")
        self.auth_headers = ({auth_header: f"Bearer {token}"} if token else {})
        self._session: requests.Session | None = None

    def _ensure_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def _request(self, method: str, url: str, operation: str, **kwargs):
        emit_log = kwargs.pop("_emit_log", True)
        if self.auth_headers:
            kwargs.setdefault("headers", {}).update(self.auth_headers)
        started_at = time.perf_counter()
        try:
            response = getattr(self._ensure_session(), method)(url, **kwargs)
        except Exception:
            record_api_call(self.metrics, logger if emit_log else None, "comfyui", operation, started_at, success=False)
            raise
        record_api_call(self.metrics, logger if emit_log else None, "comfyui", operation, started_at, success=response.ok)
        return response

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self) -> ComfyUIClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def queue_prompt(self, workflow: dict) -> str:
        response = self._request("post", f"{self.base_url}/prompt", "queue_prompt",
            json={
                "prompt": workflow,
                "client_id": self.client_id,
            },
            timeout=self.prompt_timeout_seconds,
        )
        self._raise_for_status(response, "queue prompt")
        payload = require_json_object(response.json(), context="queue_prompt")
        prompt_id = payload.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ComfyUIHTTPError("queue_prompt response missing prompt_id")
        return prompt_id

    def get_history(self, prompt_id: str) -> dict:
        response = self._request("get", f"{self.base_url}/history/{prompt_id}", "get_history",
            timeout=self.prompt_timeout_seconds,
        )
        self._raise_for_status(response, "get history")
        return require_json_object(response.json(), context="get_history")

    def get_object_info(self) -> dict:
        response = self._request("get", f"{self.base_url}/object_info", "get_object_info",
            timeout=self.prompt_timeout_seconds,
        )
        self._raise_for_status(response, "get object info")
        return response.json()

    def wait_for_completion(
        self,
        prompt_id: str,
        poll_interval: float = 1.0,
        timeout_seconds: float | None = None,
    ) -> dict:
        started_at = time.time()
        timeout_seconds = self.prompt_timeout_seconds if timeout_seconds is None else timeout_seconds

        while True:
            history = self.get_history(prompt_id)

            if prompt_id in history:
                entry = history[prompt_id]
                self._raise_for_execution_error(prompt_id, entry)
                return entry

            if time.time() - started_at > timeout_seconds:
                raise TimeoutError(f"ComfyUI prompt timed out: {prompt_id}")

            time.sleep(poll_interval)

    @staticmethod
    def _raise_for_execution_error(prompt_id: str, history_entry: dict) -> None:
        status = history_entry.get("status") or {}
        if status.get("status_str") != "error":
            return
        messages = status.get("messages") or []
        details = next(
            (
                payload
                for message_type, payload in reversed(messages)
                if message_type == "execution_error" and isinstance(payload, dict)
            ),
            {},
        )
        node_id = details.get("node_id")
        node_type = details.get("node_type")
        exception_type = details.get("exception_type")
        exception_message = details.get("exception_message")
        node = (
            f" at node {node_id} ({node_type})"
            if node_id or node_type
            else ""
        )
        exception = ": ".join(
            str(value)
            for value in (exception_type, exception_message)
            if value
        )
        suffix = f": {exception}" if exception else ""
        raise ComfyUIExecutionError(
            f"ComfyUI prompt {prompt_id} failed{node}{suffix}"
        )

    def upload_image(
        self,
        file_path: str | Path,
        subfolder: str = "",
        file_type: str = "input",
        overwrite: bool = True,
        upload_name: str | None = None,
    ) -> dict:
        """
        ComfyUI's default upload endpoint is /upload/image.
        This also works for most image inputs.

        Returned JSON usually contains:
        {
          "name": "...",
          "subfolder": "...",
          "type": "input"
        }
        """
        file_path = Path(file_path)

        with file_path.open("rb") as f:
            response = self._request("post", f"{self.base_url}/upload/image", "upload_image",
                files={"image": (upload_name or file_path.name, f)},
                data={
                    "type": file_type,
                    "subfolder": subfolder,
                    "overwrite": str(overwrite).lower(),
                },
                timeout=300,
            )

        self._raise_for_status(response, "upload image")
        return response.json()

    def upload_file_via_image_endpoint(
        self,
        file_path: str | Path,
        subfolder: str = "",
        file_type: str = "input",
        overwrite: bool = True,
        upload_name: str | None = None,
    ) -> dict:
        """
        Alias for workflows/custom nodes that accept uploaded files through ComfyUI's input folder.
        For audio/video custom nodes this may still be enough if the node expects a filename in input/.
        """
        return self.upload_image(
            file_path=file_path,
            subfolder=subfolder,
            file_type=file_type,
            overwrite=overwrite,
            upload_name=upload_name,
        )

    def input_file_exists(self, comfyui_name: str) -> bool:
        normalized = str(comfyui_name).strip().replace("\\", "/").strip("/")
        subfolder, separator, filename = normalized.rpartition("/")
        if not separator:
            filename = normalized
            subfolder = ""
        if not filename:
            return False
        response = self._request("get", f"{self.base_url}/view", "input_file_exists",
            params={"filename": filename, "subfolder": subfolder, "type": "input"},
            timeout=60,
            stream=True,
        )
        try:
            if response.status_code == 404:
                return False
            self._raise_for_status(response, "check input file")
            return True
        finally:
            response.close()

    def download_view_file(
        self,
        filename: str,
        output_path: str | Path,
        subfolder: str = "",
        file_type: str = "output",
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        response = self._request("get", f"{self.base_url}/view", "download_view_file",
            params={
                "filename": filename,
                "subfolder": subfolder,
                "type": file_type,
            },
            timeout=300,
        )
        self._raise_for_status(response, "download view file")

        output_path.write_bytes(response.content)
        return output_path

    def free_cache_and_vram(self) -> None:
        """Best-effort unload of ComfyUI models and cached CUDA memory."""
        try:
            response = self._request("post", f"{self.base_url}/free", "free_cache_and_vram", _emit_log=False,
                json={"unload_models": True, "free_memory": True},
                timeout=30,
            )
            self._raise_for_status(response, "free cache and vram")
        except Exception as exc:  # noqa: BLE001 - cache release is deliberately best-effort
            logger.debug("ComfyUI cache/VRAM release failed: %s", exc)

    def _raise_for_status(self, response: requests.Response, operation: str) -> None:
        if response.ok:
            return

        detail = self._response_detail(response)
        raise ComfyUIHTTPError(
            f"ComfyUI {operation} failed with HTTP {response.status_code} for "
            f"{redact_secrets(response.url)}: {redact_secrets(detail)}"
        ) from None

    @staticmethod
    def _response_detail(response: requests.Response) -> str:
        try:
            return json_dumps_compact(response.json())
        except ValueError:
            text = response.text.strip()
            return text if text else "<empty response body>"

    def extract_output_images(self, history_entry: dict) -> list[dict]:
        images = []
        outputs = history_entry.get("outputs", {})

        for node_id, node_output in outputs.items():
            for image in node_output.get("images", []):
                images.append({
                    "node_id": node_id,
                    "filename": image["filename"],
                    "subfolder": image.get("subfolder", ""),
                    "type": image.get("type", "output"),
                })

        return images

    def extract_output_files(self, history_entry: dict) -> list[dict]:
        """
        Some custom nodes return files/videos instead of images.
        This collects common output keys defensively.
        """
        files = []
        outputs = history_entry.get("outputs", {})

        for node_id, node_output in outputs.items():
            for key in ("files", "videos", "audio"):
                for item in node_output.get(key, []):
                    files.append({
                        "node_id": node_id,
                        "kind": key,
                        "filename": item["filename"],
                        "subfolder": item.get("subfolder", ""),
                        "type": item.get("type", "output"),
                    })

        return files
