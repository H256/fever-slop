from __future__ import annotations

from pathlib import Path
import json
import time
import uuid
import requests


class ComfyUIHTTPError(RuntimeError):
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
    ):
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id or str(uuid.uuid4())

    def queue_prompt(self, workflow: dict) -> str:
        response = requests.post(
            f"{self.base_url}/prompt",
            json={
                "prompt": workflow,
                "client_id": self.client_id,
            },
            timeout=60,
        )
        self._raise_for_status(response, "queue prompt")
        return response.json()["prompt_id"]

    def get_history(self, prompt_id: str) -> dict:
        response = requests.get(
            f"{self.base_url}/history/{prompt_id}",
            timeout=60,
        )
        self._raise_for_status(response, "get history")
        return response.json()

    def get_object_info(self) -> dict:
        response = requests.get(
            f"{self.base_url}/object_info",
            timeout=60,
        )
        self._raise_for_status(response, "get object info")
        return response.json()

    def wait_for_completion(
        self,
        prompt_id: str,
        poll_interval: float = 1.0,
        timeout_seconds: float = 1800.0,
    ) -> dict:
        started_at = time.time()

        while True:
            history = self.get_history(prompt_id)

            if prompt_id in history:
                return history[prompt_id]

            if time.time() - started_at > timeout_seconds:
                raise TimeoutError(f"ComfyUI prompt timed out: {prompt_id}")

            time.sleep(poll_interval)

    def upload_image(
        self,
        file_path: str | Path,
        subfolder: str = "",
        file_type: str = "input",
        overwrite: bool = True,
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
            response = requests.post(
                f"{self.base_url}/upload/image",
                files={"image": (file_path.name, f)},
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
        )

    def download_view_file(
        self,
        filename: str,
        output_path: str | Path,
        subfolder: str = "",
        file_type: str = "output",
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(
            f"{self.base_url}/view",
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

    def _raise_for_status(self, response: requests.Response, operation: str) -> None:
        if response.ok:
            return

        detail = self._response_detail(response)
        raise ComfyUIHTTPError(
            f"ComfyUI {operation} failed with HTTP {response.status_code} for "
            f"{response.url}: {detail}"
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
