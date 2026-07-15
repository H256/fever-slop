import json
import tempfile
import unittest
from pathlib import Path

from feverslop.adapters.comfyui_ingredients_video_backend import ComfyUIIngredientsVideoRenderBackend
from feverslop.ports.rendering import VideoRenderRequest


class FakeClient:
    def __init__(self):
        self.uploaded = []
        self.uploaded_paths = []
        self.queued_workflow = None

    def upload_image(self, path, subfolder, file_type, overwrite, upload_name=None):
        name = upload_name or Path(path).name
        self.uploaded.append(name)
        self.uploaded_paths.append(Path(path))
        return {"name": name, "subfolder": subfolder, "type": file_type}

    def upload_file_via_image_endpoint(self, path, subfolder, file_type, overwrite, upload_name):
        return {"name": upload_name, "subfolder": subfolder, "type": file_type}

    def queue_prompt(self, workflow):
        self.queued_workflow = workflow
        return "prompt-id"

    def wait_for_completion(self, prompt_id):
        return {"outputs": {"save": {"videos": [{"filename": "scene.mp4", "type": "output"}]}}}

    def download_view_file(self, filename, subfolder, file_type, output_path):
        return Path(output_path)


class FakeRenderQueue:
    def __init__(self):
        self.calls = []

    def queue_workflow_and_download_first_video(self, workflow, scene_number, output_path):
        self.calls.append({
            "workflow": workflow,
            "scene_number": scene_number,
            "output_path": Path(output_path),
        })
        return Path(output_path)


class FakePostProcessor:
    def __init__(self):
        self.trim_specs = []

    def trim_clip(self, spec):
        self.trim_specs.append(spec)
        return spec.output_file


class FakeModelResolver:
    def resolve_workflow_models(self, workflow, workflow_path=None):
        return json.loads(json.dumps(workflow))


def _build_minimal_ingredients_workflow():
    return {
        "1": {"inputs": {"image": "placeholder.png"}, "_meta": {"title": "#INGREDIENTS"}, "class_type": "LoadImage"},
        "2": {"inputs": {"text": "placeholder prompt", "clip": ["5", 0]}, "_meta": {"title": "#PROMPT_POSITIVE"}, "class_type": "CLIPTextEncode"},
        "3": {"inputs": {"text": "bad stuff", "clip": ["5", 0]}, "_meta": {"title": "#PROMPT_NEGATIVE"}, "class_type": "CLIPTextEncode"},
        "4": {"inputs": {"noise_seed": 0}, "_meta": {"title": "#SEED"}},
        "5": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
        "6": {"inputs": {"value": 1280}, "_meta": {"title": "#WIDTH"}},
        "7": {"inputs": {"value": 704}, "_meta": {"title": "#HEIGHT"}},
        "8": {"inputs": {"value": 49}, "_meta": {"title": "#FRAMES"}},
        "9": {"inputs": {"value": 24}, "_meta": {"title": "#FRAMERATE"}},
        "10": {"inputs": {"width": ["6", 0], "height": ["7", 1], "length": ["8", 0]}, "class_type": "EmptyLTXVLatentVideo"},
        "11": {"inputs": {"latent": ["10", 0]}, "class_type": "LTXVImgToVideoInplace"},
    }


def _build_audio_ingredients_workflow():
    workflow = _build_minimal_ingredients_workflow()
    workflow.update({
        "12": {"inputs": {"audio": "", "audioUI": ""}, "_meta": {"title": "#LOAD_AUDIO"}, "class_type": "LoadAudio"},
        "13": {
            "inputs": {"start_index": 0, "duration": 0, "audio": ["12", 0]},
            "_meta": {"title": "#TRIM_AUDIO"},
            "class_type": "TrimAudioDuration",
        },
    })
    return workflow


class ComfyUIIngredientsBackendTests(unittest.TestCase):
    def test_backend_patches_ingredients_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            sheet = temp / "sheet.png"
            sheet.write_bytes(b"ingredients sheet")
            workflow_path = temp / "workflow.json"
            workflow_path.write_text(json.dumps(_build_minimal_ingredients_workflow()), encoding="utf-8")

            client = FakeClient()
            backend = ComfyUIIngredientsVideoRenderBackend(
                client=client,
                workflow_path=workflow_path,
                output_dir=temp / "out",
                project_dir=temp,
                postprocess=False,
            )

            scene = {
                "scene": 1,
                "fps": 24,
                "width": 1280,
                "height": 704,
                "frame_count": 49,
                "ingredients_scene_sheet": "sheet.png",
                "ltx": {
                    "ingredients_scene_sheet_description": "Reference sheet description",
                    "ingredients_target_prompt": "### Target Description\nTarget prompt text",
                },
            }
            workflow = backend.build_workflow(scene, prompt="fallback prompt")

            patched_node = workflow["1"]
            self.assertIn("feverslop/references/", patched_node["inputs"]["image"])

    def test_prompt_concatenation_from_ltx_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            sheet = temp / "sheet.png"
            sheet.write_bytes(b"sheet")
            workflow_path = temp / "workflow.json"
            workflow_path.write_text(json.dumps(_build_minimal_ingredients_workflow()), encoding="utf-8")

            client = FakeClient()
            backend = ComfyUIIngredientsVideoRenderBackend(
                client=client,
                workflow_path=workflow_path,
                output_dir=temp / "out",
                project_dir=temp,
                postprocess=False,
            )

            scene = {
                "scene": 1,
                "fps": 24,
                "width": 1280,
                "height": 704,
                "frame_count": 49,
                "ingredients_scene_sheet": "sheet.png",
                "ltx": {
                    "ingredients_scene_sheet_description": "Description A",
                    "ingredients_target_prompt": "Target B",
                },
            }
            workflow = backend.build_workflow(scene, prompt="should not appear")

            prompt_node = workflow["2"]
            self.assertEqual(prompt_node["inputs"]["text"], "Description A\nTarget B")

    def test_prompt_fallback_to_explicit_arg(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            sheet = temp / "sheet.png"
            sheet.write_bytes(b"sheet")
            workflow_path = temp / "workflow.json"
            workflow_path.write_text(json.dumps(_build_minimal_ingredients_workflow()), encoding="utf-8")

            client = FakeClient()
            backend = ComfyUIIngredientsVideoRenderBackend(
                client=client,
                workflow_path=workflow_path,
                output_dir=temp / "out",
                project_dir=temp,
                postprocess=False,
            )

            scene = {
                "scene": 1,
                "fps": 24,
                "width": 1280,
                "height": 704,
                "frame_count": 49,
                "ingredients_scene_sheet": "sheet.png",
                "ltx": {},
            }
            workflow = backend.build_workflow(scene, prompt="explicit fallback prompt")

            prompt_node = workflow["2"]
            self.assertEqual(prompt_node["inputs"]["text"], "explicit fallback prompt")

    def test_prompt_partial_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            sheet = temp / "sheet.png"
            sheet.write_bytes(b"sheet")
            workflow_path = temp / "workflow.json"
            workflow_path.write_text(json.dumps(_build_minimal_ingredients_workflow()), encoding="utf-8")

            client = FakeClient()
            backend = ComfyUIIngredientsVideoRenderBackend(
                client=client,
                workflow_path=workflow_path,
                output_dir=temp / "out",
                project_dir=temp,
                postprocess=False,
            )

            scene = {
                "scene": 1,
                "fps": 24,
                "frame_count": 49,
                "ingredients_scene_sheet": "sheet.png",
                "ltx": {
                    "ingredients_scene_sheet_description": "Only description",
                },
            }
            workflow = backend.build_workflow(scene, prompt="fallback")

            prompt_node = workflow["2"]
            self.assertEqual(prompt_node["inputs"]["text"], "Only description")

    def test_seed_patched(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            sheet = temp / "sheet.png"
            sheet.write_bytes(b"sheet")
            workflow_path = temp / "workflow.json"
            workflow_path.write_text(json.dumps(_build_minimal_ingredients_workflow()), encoding="utf-8")

            client = FakeClient()
            backend = ComfyUIIngredientsVideoRenderBackend(
                client=client,
                workflow_path=workflow_path,
                output_dir=temp / "out",
                project_dir=temp,
                seed_offset=100000,
                postprocess=False,
            )

            scene = {
                "scene": 5,
                "fps": 24,
                "frame_count": 49,
                "ingredients_scene_sheet": "sheet.png",
                "ltx": {
                    "ingredients_scene_sheet_description": "desc",
                    "ingredients_target_prompt": "target",
                },
            }
            workflow = backend.build_workflow(scene, prompt="prompt")

            seed_node = workflow["4"]
            self.assertEqual(seed_node["inputs"]["noise_seed"], 100005)

    def test_save_video_patched(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            sheet = temp / "sheet.png"
            sheet.write_bytes(b"sheet")
            workflow_path = temp / "workflow.json"
            workflow_path.write_text(json.dumps(_build_minimal_ingredients_workflow()), encoding="utf-8")

            client = FakeClient()
            backend = ComfyUIIngredientsVideoRenderBackend(
                client=client,
                workflow_path=workflow_path,
                output_dir=temp / "out",
                project_dir=temp,
                postprocess=False,
            )

            scene = {
                "scene": 12,
                "fps": 24,
                "frame_count": 49,
                "ingredients_scene_sheet": "sheet.png",
                "ltx": {
                    "ingredients_scene_sheet_description": "desc",
                    "ingredients_target_prompt": "target",
                },
            }
            workflow = backend.build_workflow(scene, prompt="prompt")

            save_node = workflow["5"]
            self.assertEqual(save_node["inputs"]["filename_prefix"], "ltx_ingredients_raw/scene_0012")

    def test_dimensions_patched(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            sheet = temp / "sheet.png"
            sheet.write_bytes(b"sheet")
            workflow_path = temp / "workflow.json"
            workflow_path.write_text(json.dumps(_build_minimal_ingredients_workflow()), encoding="utf-8")

            client = FakeClient()
            backend = ComfyUIIngredientsVideoRenderBackend(
                client=client,
                workflow_path=workflow_path,
                output_dir=temp / "out",
                project_dir=temp,
                postprocess=False,
            )

            scene = {
                "scene": 1,
                "fps": 30,
                "width": 1920,
                "height": 1080,
                "frame_count": 90,
                "ingredients_scene_sheet": "sheet.png",
                "ltx": {
                    "ingredients_scene_sheet_description": "desc",
                    "ingredients_target_prompt": "target",
                },
            }
            workflow = backend.build_workflow(scene, prompt="prompt")

            self.assertEqual(workflow["6"]["inputs"]["value"], 1920)
            self.assertEqual(workflow["7"]["inputs"]["value"], 1080)
            self.assertEqual(workflow["8"]["inputs"]["value"], 90)
            self.assertEqual(workflow["9"]["inputs"]["value"], 30)

    def test_latent_length_patched(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            sheet = temp / "sheet.png"
            sheet.write_bytes(b"sheet")
            workflow_path = temp / "workflow.json"
            workflow_path.write_text(json.dumps(_build_minimal_ingredients_workflow()), encoding="utf-8")

            client = FakeClient()
            backend = ComfyUIIngredientsVideoRenderBackend(
                client=client,
                workflow_path=workflow_path,
                output_dir=temp / "out",
                project_dir=temp,
                postprocess=False,
            )

            scene = {
                "scene": 1,
                "fps": 24,
                "frame_count": 73,
                "ingredients_scene_sheet": "sheet.png",
                "ltx": {
                    "ingredients_scene_sheet_description": "desc",
                    "ingredients_target_prompt": "target",
                },
            }
            workflow = backend.build_workflow(scene, prompt="prompt")

            latent_node = workflow["10"]
            self.assertEqual(latent_node["inputs"]["length"], 73)

    def test_missing_ingredients_anchor_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            workflow_path = temp / "workflow.json"
            workflow_path.write_text(json.dumps({
                "2": {"inputs": {"text": ""}, "_meta": {"title": "#PROMPT_POSITIVE"}},
            }), encoding="utf-8")

            client = FakeClient()
            backend = ComfyUIIngredientsVideoRenderBackend(
                client=client,
                workflow_path=workflow_path,
                output_dir=temp / "out",
                postprocess=False,
            )

            scene = {
                "scene": 1,
                "fps": 24,
                "frame_count": 49,
                "ingredients_scene_sheet": "missing.png",
                "ltx": {},
            }
            from feverslop.errors import FeverSlopValidationError

            with self.assertRaises(FeverSlopValidationError) as ctx:
                backend.build_workflow(scene, prompt="prompt")
            self.assertIn("#INGREDIENTS", str(ctx.exception))

    def test_missing_sheet_path_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            workflow_path = temp / "workflow.json"
            workflow_path.write_text(json.dumps(_build_minimal_ingredients_workflow()), encoding="utf-8")

            client = FakeClient()
            backend = ComfyUIIngredientsVideoRenderBackend(
                client=client,
                workflow_path=workflow_path,
                output_dir=temp / "out",
                postprocess=False,
            )

            scene = {
                "scene": 3,
                "fps": 24,
                "frame_count": 49,
                "ltx": {},
            }
            from feverslop.errors import FeverSlopValidationError

            with self.assertRaises(FeverSlopValidationError) as ctx:
                backend.build_workflow(scene, prompt="prompt")
            self.assertIn("ingredients_scene_sheet", str(ctx.exception))

    def test_randomize_seed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            sheet = temp / "sheet.png"
            sheet.write_bytes(b"sheet")
            workflow_path = temp / "workflow.json"
            workflow_path.write_text(json.dumps(_build_minimal_ingredients_workflow()), encoding="utf-8")

            client = FakeClient()
            backend = ComfyUIIngredientsVideoRenderBackend(
                client=client,
                workflow_path=workflow_path,
                output_dir=temp / "out",
                project_dir=temp,
                randomize_seed=True,
                postprocess=False,
            )

            scene = {
                "scene": 1,
                "fps": 24,
                "frame_count": 49,
                "ingredients_scene_sheet": "sheet.png",
                "ltx": {
                    "ingredients_scene_sheet_description": "desc",
                    "ingredients_target_prompt": "target",
                },
            }
            workflow = backend.build_workflow(scene, prompt="prompt")

            seed_node = workflow["4"]
            self.assertIsInstance(seed_node["inputs"]["noise_seed"], int)
            self.assertGreater(seed_node["inputs"]["noise_seed"], 100000)

    def test_render_video_flow(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            sheet = temp / "sheet.png"
            sheet.write_bytes(b"sheet")
            workflow_path = temp / "workflow.json"
            workflow_path.write_text(json.dumps(_build_minimal_ingredients_workflow()), encoding="utf-8")
            (temp / "out" / "raw").mkdir(parents=True, exist_ok=True)

            render_queue = FakeRenderQueue()
            client = FakeClient()
            backend = ComfyUIIngredientsVideoRenderBackend(
                client=client,
                workflow_path=workflow_path,
                output_dir=temp / "out",
                project_dir=temp,
                postprocess=False,
                render_queue=render_queue,
                model_resolver=FakeModelResolver(),
            )

            output = backend.render_video(
                VideoRenderRequest(
                    scene={
                        "scene": 1,
                        "fps": 24,
                        "frame_count": 49,
                        "ingredients_scene_sheet": "sheet.png",
                        "ltx": {
                            "ingredients_scene_sheet_description": "desc",
                            "ingredients_target_prompt": "target",
                        },
                    },
                    scene_number=1,
                    prompt="fallback",
                    workflow_path=workflow_path,
                    output_dir=temp / "out",
                    audio_file=Path(""),
                    storyboard_dir=Path(""),
                    upload_audio=False,
                )
            )

            self.assertEqual(len(render_queue.calls), 1)
            self.assertEqual(render_queue.calls[0]["scene_number"], 1)
            self.assertEqual(output.name, "scene_0001_raw.mp4")

    def test_render_video_patches_audio_trim_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            sheet = temp / "sheet.png"
            audio = temp / "song.mp3"
            sheet.write_bytes(b"sheet")
            audio.write_bytes(b"audio")
            workflow_path = temp / "workflow.json"
            workflow_path.write_text(json.dumps(_build_audio_ingredients_workflow()), encoding="utf-8")
            (temp / "out" / "raw").mkdir(parents=True, exist_ok=True)

            render_queue = FakeRenderQueue()
            backend = ComfyUIIngredientsVideoRenderBackend(
                client=FakeClient(),
                workflow_path=workflow_path,
                output_dir=temp / "out",
                project_dir=temp,
                preroll_frames=0,
                tail_loss_frames=0,
                round_render_frames_to_8n1=False,
                postprocess=False,
                render_queue=render_queue,
                model_resolver=FakeModelResolver(),
            )

            backend.render_video(
                VideoRenderRequest(
                    scene={
                        "scene": 2,
                        "fps": 24,
                        "frame_count": 49,
                        "abs_start_seconds": 3.5,
                        "duration_seconds": 2.0,
                        "ingredients_scene_sheet": "sheet.png",
                        "ltx": {
                            "ingredients_scene_sheet_description": "desc",
                            "ingredients_target_prompt": "target",
                        },
                    },
                    scene_number=2,
                    prompt="fallback",
                    workflow_path=workflow_path,
                    output_dir=temp / "out",
                    audio_file=audio,
                    storyboard_dir=Path(""),
                )
            )

            workflow = render_queue.calls[0]["workflow"]
            audio_input = workflow["12"]["inputs"]["audio"]
            self.assertTrue(audio_input.startswith("feverslop/audio/song-"))
            self.assertEqual(f"/api/view?filename={audio_input}&type=input", workflow["12"]["inputs"]["audioUI"])
            self.assertEqual(3.5, workflow["13"]["inputs"]["start_index"])
            self.assertEqual(2.0, workflow["13"]["inputs"]["duration"])

    def test_debug_workflow_written(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            sheet = temp / "sheet.png"
            sheet.write_bytes(b"sheet")
            workflow_path = temp / "workflow.json"
            workflow_path.write_text(json.dumps(_build_minimal_ingredients_workflow()), encoding="utf-8")
            debug_dir = temp / "debug"
            (temp / "out" / "raw").mkdir(parents=True, exist_ok=True)

            render_queue = FakeRenderQueue()
            client = FakeClient()
            backend = ComfyUIIngredientsVideoRenderBackend(
                client=client,
                workflow_path=workflow_path,
                output_dir=temp / "out",
                project_dir=temp,
                debug_workflows_dir=debug_dir,
                postprocess=False,
                render_queue=render_queue,
                model_resolver=FakeModelResolver(),
            )

            backend.render_video(
                VideoRenderRequest(
                    scene={
                        "scene": 7,
                        "fps": 24,
                        "frame_count": 49,
                        "ingredients_scene_sheet": "sheet.png",
                        "ltx": {
                            "ingredients_scene_sheet_description": "desc",
                            "ingredients_target_prompt": "target",
                        },
                    },
                    scene_number=7,
                    prompt="prompt",
                    workflow_path=workflow_path,
                    output_dir=temp / "out",
                    audio_file=Path(""),
                    storyboard_dir=Path(""),
                    upload_audio=False,
                )
            )

            debug_file = debug_dir / "scene_0007_workflow.json"
            self.assertTrue(debug_file.exists())
            data = json.loads(debug_file.read_text(encoding="utf-8"))
            self.assertIn("1", data)

    def test_load_workflow_from_in_memory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            workflow_path = temp / "workflow.json"
            workflow_path.write_text(json.dumps(_build_minimal_ingredients_workflow()), encoding="utf-8")

            in_mem_workflow = _build_minimal_ingredients_workflow()
            in_mem_workflow["1"]["inputs"]["image"] = "in_memory.png"

            client = FakeClient()
            backend = ComfyUIIngredientsVideoRenderBackend(
                client=client,
                workflow_path=workflow_path,
                output_dir=temp / "out",
                workflow=in_mem_workflow,
                postprocess=False,
            )

            loaded = backend.load_workflow()
            self.assertEqual(loaded["1"]["inputs"]["image"], "in_memory.png")


if __name__ == "__main__":
    unittest.main()
