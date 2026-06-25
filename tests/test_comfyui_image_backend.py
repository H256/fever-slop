import json
import tempfile
import unittest
from pathlib import Path

from feverslop.ports.rendering import ImageRenderRequest, WorkflowAnchorConfig


class FakeComfyClient:
    def __init__(self):
        self.queued_workflow = None

    def queue_prompt(self, workflow: dict) -> str:
        self.queued_workflow = workflow
        return "prompt-id"

    def wait_for_completion(self, prompt_id: str) -> dict:
        return {"prompt_id": prompt_id}

    def extract_output_images(self, history: dict) -> list[dict]:
        return [{"filename": "out.png", "subfolder": "", "type": "output"}]

    def download_view_file(self, *, filename: str, subfolder: str, file_type: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"png")
        return output_path


def workflow_json(*, positive_input: str = "text") -> str:
    return json.dumps(
        {
            "1": {
                "class_type": "PrimitiveStringMultiline",
                "inputs": {positive_input: "old"},
                "_meta": {"title": "#PROMPT_POSITIVE"},
            },
            "2": {
                "class_type": "PrimitiveInt",
                "inputs": {"value": 1},
                "_meta": {"title": "#WIDTH"},
            },
            "3": {
                "class_type": "PrimitiveInt",
                "inputs": {"value": 1},
                "_meta": {"title": "#HEIGHT"},
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": ""},
                "_meta": {"title": "#PROMPT_NEGATIVE"},
            },
            "5": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "old", "images": ["0", 0]},
                "_meta": {"title": "#SAVE_IMAGE"},
            },
        }
    )


class ComfyUIImageBackendTests(unittest.TestCase):
    def test_patches_default_positive_prompt_input_and_dimensions(self):
        from feverslop.adapters.comfyui_rendering import ComfyUIImageBackend

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            workflow_path = temp / "workflow.json"
            workflow_path.write_text(workflow_json(), encoding="utf-8")
            client = FakeComfyClient()

            ComfyUIImageBackend(client, workflow_path, temp / "out").render_image(
                ImageRenderRequest(
                    scene={},
                    scene_number=3,
                    prompt="new prompt",
                    workflow_path=workflow_path,
                    output_dir=temp / "out",
                    width=1920,
                    height=1088,
                )
            )

        self.assertEqual("new prompt", client.queued_workflow["1"]["inputs"]["text"])
        self.assertEqual(1920, client.queued_workflow["2"]["inputs"]["value"])
        self.assertEqual(1088, client.queued_workflow["3"]["inputs"]["value"])

    def test_patches_configured_positive_prompt_input(self):
        from feverslop.adapters.comfyui_rendering import ComfyUIImageBackend

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            workflow_path = temp / "workflow.json"
            workflow_path.write_text(workflow_json(positive_input="value"), encoding="utf-8")
            client = FakeComfyClient()

            ComfyUIImageBackend(client, workflow_path, temp / "out").render_image(
                ImageRenderRequest(
                    scene={},
                    scene_number=3,
                    prompt="{raw result}",
                    workflow_path=workflow_path,
                    output_dir=temp / "out",
                    width=1280,
                    height=704,
                    anchors=WorkflowAnchorConfig(positive_prompt_input="value"),
                )
            )

        self.assertEqual("{raw result}", client.queued_workflow["1"]["inputs"]["value"])


if __name__ == "__main__":
    unittest.main()
