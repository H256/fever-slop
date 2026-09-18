from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from feverslop.adapters.comfyui_client import ComfyUIClient
from feverslop.errors import FeverSlopRenderError


@dataclass(frozen=True)
class ReplayPolicy:
    """Explicit declaration of whether an interrupted render may be re-attempted.

    Local ComfyUI renders are replayable after a restart (no billing, no
    external side effects). The policy is declared, not assumed, so a future
    execution target must state its own replay behavior.
    """

    replayable: bool
    reason: str


# Local ComfyUI is the only execution target in scope; it is replayable.
COMFYUI_REPLAY_POLICY = ReplayPolicy(replayable=True, reason="local ComfyUI, no billing")


class ComfyUIRenderQueue:
    def __init__(self, client: ComfyUIClient):
        self.client = client

    @property
    def replay_policy(self) -> ReplayPolicy:
        """The declared replay policy for this ComfyUI execution target."""
        return COMFYUI_REPLAY_POLICY

    def queue_workflow_and_download_first_video(
        self,
        workflow: dict,
        *,
        scene_number: int,
        output_path: Path,
    ) -> Path:
        prompt_id = self.client.queue_prompt(workflow)
        history = self.client.wait_for_completion(prompt_id)

        videos = self.extract_output_videos(history)
        if not videos:
            raise FeverSlopRenderError(f"No video output for scene {scene_number}")

        first = videos[0]
        return self.client.download_view_file(
            filename=first["filename"],
            subfolder=first.get("subfolder", ""),
            file_type=first.get("type", "output"),
            output_path=output_path,
        )

    @staticmethod
    def extract_output_videos(history_entry: dict) -> list[dict]:
        videos = []
        outputs = history_entry.get("outputs", {})
        for node_id, node_output in outputs.items():
            for key in ("videos", "gifs", "files", "images"):
                for item in node_output.get(key, []):
                    filename = item.get("filename")
                    if filename and filename.lower().endswith((".mp4", ".mov", ".mkv", ".webm")):
                        videos.append({
                            "node_id": node_id,
                            "filename": filename,
                            "subfolder": item.get("subfolder", ""),
                            "type": item.get("type", "output"),
                        })
        return videos
