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
        },
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
                ),
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
                ),
            )

        self.assertEqual("{raw result}", client.queued_workflow["1"]["inputs"]["value"])

    def test_patches_matching_positive_prompt_input_when_titles_are_duplicated(self):
        from feverslop.adapters.comfyui_rendering import ComfyUIImageBackend

        workflow = {
            "1": {
                "class_type": "PrimitiveStringMultiline",
                "inputs": {"value": "old value"},
                "_meta": {"title": "#PROMPT_POSITIVE"},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "old text"},
                "_meta": {"title": "#PROMPT_POSITIVE"},
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": ""},
                "_meta": {"title": "#PROMPT_NEGATIVE"},
            },
            "4": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "old", "images": ["0", 0]},
                "_meta": {"title": "#SAVE_IMAGE"},
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            workflow_path = temp / "workflow.json"
            workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
            client = FakeComfyClient()

            ComfyUIImageBackend(client, workflow_path, temp / "out").render_image(
                ImageRenderRequest(
                    scene={},
                    scene_number=3,
                    prompt="new text",
                    workflow_path=workflow_path,
                    output_dir=temp / "out",
                    anchors=WorkflowAnchorConfig(positive_prompt_input="text"),
                ),
            )

        self.assertEqual({"value": "old value"}, client.queued_workflow["1"]["inputs"])
        self.assertEqual("new text", client.queued_workflow["2"]["inputs"]["text"])

    def test_patches_dimensions_anchor_when_width_height_anchors_are_absent(self):
        from feverslop.adapters.comfyui_rendering import ComfyUIImageBackend

        workflow = {
            "1": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": ""},
                "_meta": {"title": "#PROMPT_POSITIVE"},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": ""},
                "_meta": {"title": "#PROMPT_NEGATIVE"},
            },
            "3": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 1280, "height": 720, "batch_size": 1},
                "_meta": {"title": "#DIMENSIONS"},
            },
            "4": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "old", "images": ["0", 0]},
                "_meta": {"title": "#SAVE_IMAGE"},
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            workflow_path = temp / "workflow.json"
            workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
            client = FakeComfyClient()

            ComfyUIImageBackend(client, workflow_path, temp / "out").render_image(
                ImageRenderRequest(
                    scene={},
                    scene_number=1,
                    prompt="portrait",
                    workflow_path=workflow_path,
                    output_dir=temp / "out",
                    width=832,
                    height=1216,
                ),
            )

        self.assertEqual(832, client.queued_workflow["3"]["inputs"]["width"])
        self.assertEqual(1216, client.queued_workflow["3"]["inputs"]["height"])

    def test_patches_seed_inputs_even_without_seed_anchor(self):
        from feverslop.adapters.comfyui_rendering import ComfyUIImageBackend

        workflow = {
            "1": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": ""},
                "_meta": {"title": "#PROMPT_POSITIVE"},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": ""},
                "_meta": {"title": "#PROMPT_NEGATIVE"},
            },
            "3": {
                "class_type": "KSampler",
                "inputs": {"seed": 672800148947068},
                "_meta": {"title": "KSampler"},
            },
            "4": {
                "class_type": "RandomNoise",
                "inputs": {"noise_seed": 192774551144773},
                "_meta": {"title": "RandomNoise"},
            },
            "5": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "old", "images": ["0", 0]},
                "_meta": {"title": "#SAVE_IMAGE"},
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            workflow_path = temp / "workflow.json"
            workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
            client = FakeComfyClient()

            ComfyUIImageBackend(client, workflow_path, temp / "out").render_image(
                ImageRenderRequest(
                    scene={},
                    scene_number=4,
                    prompt="portrait",
                    workflow_path=workflow_path,
                    output_dir=temp / "out",
                ),
            )

        self.assertEqual(100004, client.queued_workflow["3"]["inputs"]["seed"])
        self.assertEqual(100004, client.queued_workflow["4"]["inputs"]["noise_seed"])


if __name__ == "__main__":
    unittest.main()
