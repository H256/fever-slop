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

    def test_backend_rejects_scene_without_actor_reference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            location = temp / "location.png"
            location.write_bytes(b"location")
            workflow = temp / "workflow.json"
            workflow.write_text("{}", encoding="utf-8")
            backend = ComfyUIMSRVideoRenderBackend(client=FakeClient(), workflow_path=workflow, output_dir=temp / "out")

            with self.assertRaisesRegex(ValueError, "at least 1 actor"):
                backend.build_workflow(
                    {
                        "scene": 1,
                        "references": {
                            "actor_sheet_paths": [],
                            "location_sheet_path": str(location),
                        },
                    },
                    prompt="prompt",
                )

    def test_backend_prefers_single_msr_images_over_review_sheets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            actor_sheet = temp / "actor_sheet.png"
            actor_msr = temp / "actor_hero.png"
            location_sheet = temp / "location_sheet.png"
            location_msr = temp / "location_hero.png"
            for path in (actor_sheet, actor_msr, location_sheet, location_msr):
                path.write_bytes(path.name.encode("utf-8"))
            workflow = temp / "workflow.json"
            workflow.write_text(
                json.dumps({
                    "1": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_ACTOR_1"}},
                    "2": {"inputs": {"image": ""}, "_meta": {"title": "#MSR_BACKGROUND"}},
                    "3": {"inputs": {"text": ""}, "_meta": {"title": "#PROMPT"}},
                    "4": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
                }),
                encoding="utf-8",
            )
            client = FakeClient()
            backend = ComfyUIMSRVideoRenderBackend(client=client, workflow_path=workflow, output_dir=temp / "out")

            patched = backend.build_workflow(
                {
                    "scene": 3,
                    "references": {
                        "actor_sheet_paths": [str(actor_sheet)],
                        "location_sheet_path": str(location_sheet),
                        "actor_msr_paths": [str(actor_msr)],
                        "location_msr_path": str(location_msr),
                    },
                },
                prompt="prompt",
            )

            self.assertEqual("feverslop/references/actor_hero.png", patched["1"]["inputs"]["image"])
            self.assertEqual("feverslop/references/location_hero.png", patched["2"]["inputs"]["image"])
