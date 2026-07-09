from __future__ import annotations

from pathlib import Path

from feverslop.adapters.comfyui_client import ComfyUIClient


class ComfyUIRenderQueue:
    def __init__(self, client: ComfyUIClient):
        self.client = client

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
            raise RuntimeError(f"No video output for scene {scene_number}")

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
            for key in ("videos", "gifs", "files"):
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
