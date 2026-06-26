import json
import tempfile
import unittest
from pathlib import Path

from feverslop.adapters.comfyui_msr_video_backend import ComfyUIMSRVideoRenderBackend
from feverslop.ports.rendering import VideoRenderRequest


class FakeClient:
    def __init__(self):
        self.uploaded = []
        self.queued_workflow = None

    def upload_image(self, path, subfolder, file_type, overwrite):
        self.uploaded.append(Path(path).name)
        return {"name": Path(path).name, "subfolder": subfolder, "type": file_type}

    def upload_file_via_image_endpoint(self, path, subfolder, file_type, overwrite, upload_name):
        return {"name": upload_name, "subfolder": subfolder, "type": file_type}

    def queue_prompt(self, workflow):
        self.queued_workflow = workflow
        return "prompt-id"

    def wait_for_completion(self, prompt_id):
        return {"outputs": {"save": {"videos": [{"filename": "scene.mp4", "type": "output"}]}}}

    def download_view_file(self, filename, subfolder, file_type, output_path):
        return Path(output_path)


class LTXMSRVideoBackendTests(unittest.TestCase):
    def test_backend_patches_msr_references_and_prompt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            actor = temp / "actor.png"
            location = temp / "location.png"
            actor.write_bytes(b"actor")
            location.write_bytes(b"location")
            workflow = temp / "workflow.json"
            workflow.write_text(
                json.dumps({
                    "1": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_ACTOR_1"}},
                    "2": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_BACKGROUND"}},
                    "3": {"inputs": {"frame_count": 17}, "_meta": {"title": "#MSR_FRAME_COUNT"}},
                    "4": {"inputs": {"text": ""}, "_meta": {"title": "#PROMPT"}},
                    "5": {"inputs": {"noise_seed": 0}, "_meta": {"title": "#SEED"}},
                    "6": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
                }),
                encoding="utf-8",
            )
            client = FakeClient()
            backend = ComfyUIMSRVideoRenderBackend(client=client, workflow_path=workflow, output_dir=temp / "out")

            output = backend.render_video(
                VideoRenderRequest(
                    scene={
                        "scene": 7,
                        "fps": 24,
                        "width": 1280,
                        "height": 704,
                        "ltx": {"original_style_i2v_prompt": "video prompt"},
                        "references": {
                            "actor_sheet_paths": [str(actor)],
                            "location_sheet_path": str(location),
                        },
                    },
                    scene_number=7,
                    prompt="video prompt",
                    workflow_path=workflow,
                    output_dir=temp / "out",
                    audio_file=temp / "song.mp3",
                    storyboard_dir=temp,
                )
            )

            self.assertEqual(temp / "out" / "scene_0007_raw.mp4", output)
            self.assertEqual("feverslop/references/actor.png", client.queued_workflow["1"]["inputs"]["image"])
            self.assertEqual("feverslop/references/location.png", client.queued_workflow["2"]["inputs"]["image"])
            self.assertEqual(17, client.queued_workflow["3"]["inputs"]["frame_count"])
            self.assertEqual("video prompt", client.queued_workflow["4"]["inputs"]["text"])
            self.assertEqual(100007, client.queued_workflow["5"]["inputs"]["noise_seed"])
