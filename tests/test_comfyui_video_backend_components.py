import unittest
import hashlib
import json
import tempfile
from pathlib import Path


class FakeComfyUIClient:
    def __init__(self):
        self.audio_uploads = []
        self.image_uploads = []

    def upload_file_via_image_endpoint(self, path, *, subfolder, file_type, overwrite, upload_name=None):
        self.audio_uploads.append((Path(path), subfolder, file_type, overwrite, upload_name))
        return {"name": upload_name or Path(path).name, "subfolder": subfolder}

    def upload_image(self, path, *, subfolder, file_type, overwrite, upload_name=None):
        self.image_uploads.append((Path(path), subfolder, file_type, overwrite, upload_name))
        return {"name": upload_name or Path(path).name, "subfolder": subfolder}


class ComfyUIVideoAssetUploaderTests(unittest.TestCase):
    def test_upload_audio_uses_comfyui_image_endpoint_contract(self):
        from feverslop.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader

        client = FakeComfyUIClient()
        uploader = ComfyUIVideoAssetUploader(client)

        name = uploader.resolve_audio_name(
            Path("song.mp3"),
            upload_audio=True,
            uploaded_audio_name=None,
        )

        self.assertEqual("feverslop/audio/song.mp3", name)
        self.assertEqual(
            [(Path("song.mp3"), "feverslop/audio", "input", True, "song.mp3")],
            client.audio_uploads,
        )

    def test_existing_audio_upload_uses_content_addressed_name(self):
        from feverslop.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader

        with tempfile.TemporaryDirectory() as temp_dir:
            audio = Path(temp_dir) / "song.mp3"
            audio.write_bytes(b"audio")
            digest = hashlib.sha256(b"audio").hexdigest()[:12]
            client = FakeComfyUIClient()
            uploader = ComfyUIVideoAssetUploader(client)

            name = uploader.resolve_audio_name(
                audio,
                upload_audio=True,
                uploaded_audio_name=None,
            )

        self.assertEqual(f"feverslop/audio/song-{digest}.mp3", name)
        self.assertEqual(
            [(audio, "feverslop/audio", "input", True, f"song-{digest}.mp3")],
            client.audio_uploads,
        )

    def test_audio_upload_can_be_skipped_with_uploaded_name_or_file_name(self):
        from feverslop.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader

        uploader = ComfyUIVideoAssetUploader(FakeComfyUIClient())

        self.assertEqual(
            "already/uploaded.mp3",
            uploader.resolve_audio_name(
                Path("song.mp3"),
                upload_audio=False,
                uploaded_audio_name="already/uploaded.mp3",
            ),
        )
        self.assertEqual(
            "song.mp3",
            uploader.resolve_audio_name(
                Path("song.mp3"),
                upload_audio=False,
                uploaded_audio_name=None,
            ),
        )

    def test_upload_startframe_uses_storyboard_subfolder_contract(self):
        from feverslop.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader

        client = FakeComfyUIClient()
        uploader = ComfyUIVideoAssetUploader(client)

        name = uploader.resolve_startframe_name(
            Path("scene_0001.png"),
            upload_startframes=True,
        )

        self.assertEqual("feverslop/storyboard/scene_0001.png", name)
        self.assertEqual(
            [(Path("scene_0001.png"), "feverslop/storyboard", "input", True, None)],
            client.image_uploads,
        )

    def test_startframe_upload_can_be_skipped(self):
        from feverslop.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader

        uploader = ComfyUIVideoAssetUploader(FakeComfyUIClient())

        self.assertEqual(
            "scene_0001.png",
            uploader.resolve_startframe_name(
                Path("scene_0001.png"),
                upload_startframes=False,
            ),
        )

    def test_malformed_upload_response_raises_clear_error(self):
        from feverslop.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader

        with self.assertRaisesRegex(ValueError, "Unexpected ComfyUI upload response"):
            ComfyUIVideoAssetUploader.comfy_path_from_upload({"subfolder": "x"})


class LTXWorkflowPatcherTests(unittest.TestCase):
    def _workflow_with_titles(self, titles: list[str]) -> str:
        return "{" + ",".join(
            f'"{index}": {{"inputs": {{}}, "_meta": {{"title": "{title}"}}}}'
            for index, title in enumerate(titles, start=1)
        ) + "}"

    def _settings(self, temp: Path, **overrides):
        from feverslop.adapters.ltx_workflow_patcher import LTXWorkflowSettings

        relay_path = temp / "relay.json"
        single_path = temp / "single.json"
        relay_path.write_text(
            self._workflow_with_titles(
                [
                    "#WIDTH",
                    "#HEIGHT",
                    "#LOAD_AUDIO",
                    "#TRIM_AUDIO",
                    "#STARTFRAME",
                    "#FRAMES",
                    "#FRAMERATE",
                    "#SEED",
                    "#PROMPT_RELAY",
                    "#SAVE_VIDEO",
                    "#LORA_1",
                ]
            ),
            encoding="utf-8",
        )
        single_path.write_text(
            self._workflow_with_titles(
                [
                    "#WIDTH",
                    "#HEIGHT",
                    "#LOAD_AUDIO",
                    "#TRIM_AUDIO",
                    "#STARTFRAME",
                    "#FRAMES",
                    "#FRAMERATE",
                    "#SEED",
                    "#PROMPT_POSITIVE",
                    "#SAVE_VIDEO",
                    "#LORA_1",
                ]
            ),
            encoding="utf-8",
        )
        values = dict(
            ltx_workflow_path=relay_path,
            single_prompt_workflow_path=single_path,
            render_mode="relay",
            width_node_title="#WIDTH",
            height_node_title="#HEIGHT",
            load_audio_node_title="#LOAD_AUDIO",
            trim_audio_node_title="#TRIM_AUDIO",
            startframe_node_title="#STARTFRAME",
            frames_node_title="#FRAMES",
            framerate_node_title="#FRAMERATE",
            seed_node_title="#SEED",
            prompt_relay_node_title="#PROMPT_RELAY",
            single_prompt_node_title="#PROMPT",
            single_prompt_input_name="text",
            save_video_node_title="#SAVE_VIDEO",
            character_lora_node_title="#LORA_1",
            character_lora_strength=1.0,
            lora_1_enabled=False,
            lora_1_name="",
            lora_1_strength_model=1.0,
            lora_1_strength_clip=1.0,
            lora_1_strengths_explicit=False,
            lora_1_node_title="#LORA_1",
            randomize_seed=False,
            seed_offset=100000,
            segment_length_mode="frames_minus_one",
            debug_workflows_dir=None,
        )
        values.update(overrides)
        return LTXWorkflowSettings(**values)

    def test_relay_validation_requires_prompt_relay_anchor(self):
        from feverslop.adapters.ltx_workflow_patcher import LTXWorkflowPatcher

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            settings = self._settings(temp)
            settings.ltx_workflow_path.write_text(
                self._workflow_with_titles(
                    [
                        "#WIDTH",
                        "#HEIGHT",
                        "#LOAD_AUDIO",
                        "#TRIM_AUDIO",
                        "#STARTFRAME",
                        "#FRAMES",
                        "#FRAMERATE",
                        "#SEED",
                        "#SAVE_VIDEO",
                    ]
                ),
                encoding="utf-8",
            )

            from feverslop.errors import FeverSlopWorkflowError

            with self.assertRaisesRegex(FeverSlopWorkflowError, "#PROMPT_RELAY"):
                LTXWorkflowPatcher(settings).validate_workflow(mode="relay")

    def test_single_prompt_validation_accepts_prompt_positive_fallback(self):
        from feverslop.adapters.ltx_workflow_patcher import LTXWorkflowPatcher

        with tempfile.TemporaryDirectory() as temp_dir:
            patcher = LTXWorkflowPatcher(self._settings(Path(temp_dir)))

            patcher.validate_workflow(mode="single_prompt")

    def test_validation_accepts_empty_audio_workflow_without_audio_anchors(self):
        from feverslop.adapters.ltx_workflow_patcher import LTXWorkflowPatcher

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            settings = self._settings(temp)
            settings.ltx_workflow_path.write_text(
                json.dumps(
                    {
                        "1": {"inputs": {"value": 1280}, "_meta": {"title": "#WIDTH"}},
                        "2": {"inputs": {"value": 704}, "_meta": {"title": "#HEIGHT"}},
                        "3": {"inputs": {"image": ""}, "_meta": {"title": "#STARTFRAME"}},
                        "4": {"inputs": {"value": 24}, "_meta": {"title": "#FRAMES"}},
                        "5": {"inputs": {"value": 24}, "_meta": {"title": "#FRAMERATE"}},
                        "6": {"inputs": {"noise_seed": 0}, "_meta": {"title": "#SEED"}},
                        "7": {
                            "inputs": {"global_prompt": "", "local_prompts": "", "segment_lengths": ""},
                            "_meta": {"title": "#PROMPT_RELAY"},
                        },
                        "8": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
                        "9": {"inputs": {}, "class_type": "LTXVEmptyLatentAudio"},
                    }
                ),
                encoding="utf-8",
            )

            LTXWorkflowPatcher(settings).validate_workflow(mode="relay")

    def test_empty_audio_workflow_build_skips_removed_audio_anchors(self):
        from feverslop.adapters.ltx_workflow_patcher import LTXWorkflowPatcher
        from feverslop.domain.ltx_rendering import AudioWindowSpec

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            settings = self._settings(temp, render_mode="single_prompt")
            settings.single_prompt_workflow_path.write_text(
                json.dumps(
                    {
                        "1": {"inputs": {"value": 1280}, "_meta": {"title": "#WIDTH"}},
                        "2": {"inputs": {"value": 704}, "_meta": {"title": "#HEIGHT"}},
                        "3": {"inputs": {"image": ""}, "_meta": {"title": "#STARTFRAME"}},
                        "4": {"inputs": {"value": 24}, "_meta": {"title": "#FRAMES"}},
                        "5": {"inputs": {"value": 24}, "_meta": {"title": "#FRAMERATE"}},
                        "6": {"inputs": {"noise_seed": 0}, "_meta": {"title": "#SEED"}},
                        "7": {"inputs": {"text": ""}, "_meta": {"title": "#PROMPT_POSITIVE"}},
                        "8": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
                        "9": {"inputs": {}, "class_type": "LTXVEmptyLatentAudio"},
                    }
                ),
                encoding="utf-8",
            )

            workflow = LTXWorkflowPatcher(settings).build_workflow(
                scene={
                    "scene": 1,
                    "fps": 24,
                    "width": 1280,
                    "height": 704,
                    "ltx": {"original_style_i2v_prompt": "prompt"},
                },
                comfy_audio_name="song.mp3",
                comfy_startframe_name="scene_0001.png",
                rolling=AudioWindowSpec(
                    scene_frame_count=24,
                    render_frame_count=24,
                    trim_front_frames=0,
                    tail_loss_frames=0,
                    fps=24,
                    audio_start_seconds=0,
                    audio_duration_seconds=1,
                ),
            )

            titles = {node.get("_meta", {}).get("title") for node in workflow.values()}
            self.assertNotIn("#LOAD_AUDIO", titles)
            self.assertNotIn("#TRIM_AUDIO", titles)
            startframe_node = next(node for node in workflow.values() if node["_meta"]["title"] == "#STARTFRAME")
            self.assertEqual("scene_0001.png", startframe_node["inputs"]["image"])

    def test_workflow_loading_accepts_utf8_bom(self):
        from feverslop.adapters.ltx_workflow_patcher import LTXWorkflowPatcher

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = self._settings(Path(temp_dir))
            workflow_text = settings.ltx_workflow_path.read_text(encoding="utf-8")
            settings.ltx_workflow_path.write_text(f"\ufeff{workflow_text}", encoding="utf-8")

            patcher = LTXWorkflowPatcher(settings)

            patcher.validate_workflow(mode="relay")
            self.assertIn("1", patcher.load_workflow(mode="relay"))

    def test_lora_enabled_requires_lora_anchor(self):
        from feverslop.adapters.ltx_workflow_patcher import LTXWorkflowPatcher

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            settings = self._settings(temp, lora_1_enabled=True, lora_1_name="characters/test.safetensors")
            settings.ltx_workflow_path.write_text(
                self._workflow_with_titles(
                    [
                        "#WIDTH",
                        "#HEIGHT",
                        "#LOAD_AUDIO",
                        "#TRIM_AUDIO",
                        "#STARTFRAME",
                        "#FRAMES",
                        "#FRAMERATE",
                        "#SEED",
                        "#PROMPT_RELAY",
                        "#SAVE_VIDEO",
                    ]
                ),
                encoding="utf-8",
            )

            from feverslop.errors import FeverSlopWorkflowError

            with self.assertRaisesRegex(FeverSlopWorkflowError, "#LORA_1"):
                LTXWorkflowPatcher(settings).validate_workflow(mode="relay")

    def test_single_prompt_build_workflow_patches_original_style_prompt(self):
        from feverslop.adapters.ltx_workflow_patcher import LTXWorkflowPatcher
        from feverslop.domain.ltx_rendering import AudioWindowSpec

        with tempfile.TemporaryDirectory() as temp_dir:
            patcher = LTXWorkflowPatcher(self._settings(Path(temp_dir), render_mode="single_prompt"))

            workflow = patcher.build_workflow(
                scene={
                    "scene": 1,
                    "fps": 24,
                    "width": 1280,
                    "height": 704,
                    "ltx": {
                        "base_prompt": "base prompt",
                        "original_style_i2v_prompt": "original style prompt",
                    },
                },
                comfy_audio_name="song.mp3",
                comfy_startframe_name="scene_0001.png",
                rolling=AudioWindowSpec(
                    scene_frame_count=24,
                    render_frame_count=24,
                    trim_front_frames=0,
                    tail_loss_frames=0,
                    fps=24,
                    audio_start_seconds=0,
                    audio_duration_seconds=1,
                ),
            )

            prompt_node = next(node for node in workflow.values() if node["_meta"]["title"] == "#PROMPT_POSITIVE")
            self.assertEqual("original style prompt", prompt_node["inputs"]["text"])

    def test_lora_explicit_strengths_patch_workflow_defaults(self):
        from feverslop.adapters.ltx_workflow_patcher import LTXWorkflowPatcher
        from feverslop.domain.ltx_rendering import AudioWindowSpec

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            settings = self._settings(
                temp,
                render_mode="single_prompt",
                lora_1_strength_model=0.0,
                lora_1_strength_clip=0.0,
                lora_1_strengths_explicit=True,
            )
            settings.single_prompt_workflow_path.write_text(
                json.dumps(
                    {
                        "1": {"inputs": {"value": 1280}, "_meta": {"title": "#WIDTH"}},
                        "2": {"inputs": {"value": 704}, "_meta": {"title": "#HEIGHT"}},
                        "3": {"inputs": {"audio": "", "audioUI": ""}, "_meta": {"title": "#LOAD_AUDIO"}},
                        "4": {"inputs": {"start_index": 0, "duration": 1}, "_meta": {"title": "#TRIM_AUDIO"}},
                        "5": {"inputs": {"image": ""}, "_meta": {"title": "#STARTFRAME"}},
                        "6": {"inputs": {"value": 24}, "_meta": {"title": "#FRAMES"}},
                        "7": {"inputs": {"value": 24}, "_meta": {"title": "#FRAMERATE"}},
                        "8": {"inputs": {"noise_seed": 0}, "_meta": {"title": "#SEED"}},
                        "9": {"inputs": {"text": ""}, "_meta": {"title": "#PROMPT_POSITIVE"}},
                        "10": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
                        "11": {
                            "inputs": {"lora_name": "workflow-default.safetensors", "strength_model": 0.85},
                            "_meta": {"title": "#LORA_1"},
                        },
                    }
                ),
                encoding="utf-8",
            )

            workflow = LTXWorkflowPatcher(settings).build_workflow(
                scene={
                    "scene": 1,
                    "fps": 24,
                    "width": 1280,
                    "height": 704,
                    "ltx": {"original_style_i2v_prompt": "prompt"},
                },
                comfy_audio_name="song.mp3",
                comfy_startframe_name="scene_0001.png",
                rolling=AudioWindowSpec(
                    scene_frame_count=24,
                    render_frame_count=24,
                    trim_front_frames=0,
                    tail_loss_frames=0,
                    fps=24,
                    audio_start_seconds=0,
                    audio_duration_seconds=1,
                ),
            )

            lora_node = next(node for node in workflow.values() if node["_meta"]["title"] == "#LORA_1")
            self.assertEqual("workflow-default.safetensors", lora_node["inputs"]["lora_name"])
            self.assertEqual(0.0, lora_node["inputs"]["strength_model"])

    def test_debug_workflow_file_is_written(self):
        from feverslop.adapters.ltx_workflow_patcher import LTXWorkflowPatcher
        from feverslop.domain.ltx_rendering import AudioWindowSpec

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            debug_dir = temp / "debug"
            patcher = LTXWorkflowPatcher(self._settings(temp, render_mode="single_prompt", debug_workflows_dir=debug_dir))

            patcher.build_workflow(
                scene={
                    "scene": 1,
                    "fps": 24,
                    "width": 1280,
                    "height": 704,
                    "ltx": {"original_style_i2v_prompt": "prompt"},
                },
                comfy_audio_name="song.mp3",
                comfy_startframe_name="scene_0001.png",
                rolling=AudioWindowSpec(
                    scene_frame_count=24,
                    render_frame_count=24,
                    trim_front_frames=0,
                    tail_loss_frames=0,
                    fps=24,
                    audio_start_seconds=0,
                    audio_duration_seconds=1,
                ),
            )

            self.assertTrue((debug_dir / "scene_0001_workflow.json").exists())

    def test_lora_patching_is_delegated_to_lora_workflow_patcher(self):
        from feverslop.adapters.lora_workflow_patcher import LoraWorkflowPatcher
        from feverslop.adapters.ltx_workflow_patcher import LTXWorkflowPatcher, ResolvedLoraConfig
        from feverslop.adapters.workflow_patcher import WorkflowPatcher

        with tempfile.TemporaryDirectory() as temp_dir:
            ltx = LTXWorkflowPatcher(
                self._settings(
                    Path(temp_dir),
                    loras=(
                        ResolvedLoraConfig(
                            index=1,
                            enabled=True,
                            name="characters/first.safetensors",
                            strength_model=0.8,
                            strength_clip=0.6,
                            name_explicit=True,
                            strength_model_explicit=True,
                            strength_clip_explicit=True,
                        ),
                    ),
                )
            )
            patcher = WorkflowPatcher(
                {
                    "1": {
                        "inputs": {"lora_name": "", "strength_model": 1.0, "strength_clip": 1.0},
                        "class_type": "LoraLoader",
                        "_meta": {"title": "#LORA_1"},
                    }
                }
            )

            ltx.patch_lora_inputs(patcher)

            self.assertIsInstance(ltx.lora_patcher, LoraWorkflowPatcher)
            self.assertEqual("characters/first.safetensors", patcher.get()["1"]["inputs"]["lora_name"])


class FakeQueueClient:
    def __init__(self, history):
        self.history = history
        self.queued_workflow = None
        self.downloads = []

    def queue_prompt(self, workflow):
        self.queued_workflow = workflow
        return "prompt-id"

    def wait_for_completion(self, prompt_id):
        self.prompt_id = prompt_id
        return self.history

    def download_view_file(self, *, filename, subfolder, file_type, output_path):
        self.downloads.append((filename, subfolder, file_type, Path(output_path)))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake video")
        return Path(output_path)


class ComfyUIRenderQueueTests(unittest.TestCase):
    def test_extract_output_videos_accepts_supported_video_extensions(self):
        from feverslop.adapters.comfyui_render_queue import ComfyUIRenderQueue

        videos = ComfyUIRenderQueue.extract_output_videos(
            {
                "outputs": {
                    "1": {
                        "videos": [{"filename": "a.mp4", "subfolder": "v", "type": "output"}],
                        "gifs": [{"filename": "b.mov", "subfolder": "", "type": "output"}],
                        "files": [
                            {"filename": "c.mkv", "subfolder": "", "type": "output"},
                            {"filename": "d.webm", "subfolder": "", "type": "output"},
                            {"filename": "note.txt", "subfolder": "", "type": "output"},
                        ],
                    }
                }
            }
        )

        self.assertEqual(["a.mp4", "b.mov", "c.mkv", "d.webm"], [item["filename"] for item in videos])

    def test_queue_download_raises_when_history_has_no_video_output(self):
        from feverslop.adapters.comfyui_render_queue import ComfyUIRenderQueue

        queue = ComfyUIRenderQueue(FakeQueueClient({"outputs": {"1": {"files": [{"filename": "note.txt"}]}}}))

        from feverslop.errors import FeverSlopRenderError

        with self.assertRaisesRegex(FeverSlopRenderError, "No video output for scene 7"):
            queue.queue_workflow_and_download_first_video(
                {"workflow": True},
                scene_number=7,
                output_path=Path("raw/scene_0007_raw.mp4"),
            )

    def test_queue_downloads_first_video_to_requested_path(self):
        from feverslop.adapters.comfyui_render_queue import ComfyUIRenderQueue

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "raw" / "scene_0001_raw.mp4"
            client = FakeQueueClient(
                {
                    "outputs": {
                        "save": {
                            "videos": [
                                {"filename": "scene_0001.mp4", "subfolder": "ltx_raw", "type": "output"}
                            ]
                        }
                    }
                }
            )
            queue = ComfyUIRenderQueue(client)

            rendered = queue.queue_workflow_and_download_first_video(
                {"workflow": True},
                scene_number=1,
                output_path=output,
            )

            self.assertEqual(output, rendered)
            self.assertEqual({"workflow": True}, client.queued_workflow)
            self.assertEqual("prompt-id", client.prompt_id)
            self.assertEqual(
                [("scene_0001.mp4", "ltx_raw", "output", output)],
                client.downloads,
            )


class FakeAssetUploader:
    def __init__(self):
        self.audio_calls = []
        self.startframe_calls = []

    def resolve_audio_name(self, audio_file, *, upload_audio, uploaded_audio_name):
        self.audio_calls.append((Path(audio_file), upload_audio, uploaded_audio_name))
        return "audio/comfy-song.mp3"

    def resolve_startframe_name(self, startframe_path, *, upload_startframes):
        self.startframe_calls.append((Path(startframe_path), upload_startframes))
        return "storyboard/comfy-scene.png"


class FakeWorkflowPatcher:
    def __init__(self):
        self.workflow_calls = []

    def build_workflow(self, *, scene, comfy_audio_name, comfy_startframe_name, rolling):
        self.workflow_calls.append((scene["scene"], comfy_audio_name, comfy_startframe_name, rolling.render_frame_count))
        return {"scene": scene["scene"]}

    def load_workflow(self, mode="relay"):
        return {"mode": mode}

    def validate_workflow(self, mode="relay"):
        self.validated_mode = mode

    def render_mode_for_scene(self, scene):
        return "single_prompt"

    def workflow_path_for_mode(self, mode):
        return Path(f"{mode}.json")

    def patch_lora_inputs(self, patcher):
        self.patch_lora_called = True

    def patch_prompt_inputs(self, **kwargs):
        self.patch_prompt_called = True

    def seed_for_scene(self, scene_number):
        return 100000 + scene_number

    def build_prompt_relay_payload(self, *, scene, render_frame_count, trim_front_frames, tail_loss_frames):
        return "global", "local", "23"


class FakeRenderQueue:
    def __init__(self):
        self.calls = []

    def queue_workflow_and_download_first_video(self, workflow, *, scene_number, output_path):
        self.calls.append((workflow, scene_number, Path(output_path)))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"raw")
        return Path(output_path)


class FakePostprocessor:
    def __init__(self):
        self.trim_specs = []
        self.concat_calls = []
        self.manifest_calls = []

    def trim_clip(self, spec):
        self.trim_specs.append(spec)
        spec.output_file.parent.mkdir(parents=True, exist_ok=True)
        spec.output_file.write_bytes(b"final")
        return spec.output_file

    def write_concat_list(self, video_files, output_file):
        self.concat_calls.append((list(video_files), Path(output_file)))
        return Path(output_file)

    def write_manifest(self, entries, output_file):
        self.manifest_calls.append((list(entries), Path(output_file)))
        return Path(output_file)


class FakeModelResolver:
    def __init__(self):
        self.calls = []

    def resolve_workflow_models(self, workflow, workflow_path=None):
        self.calls.append((workflow, Path(workflow_path) if workflow_path is not None else None))
        patched = dict(workflow)
        patched["resolved"] = {"inputs": {"ok": True}}
        return patched


class ComfyUIImageBackendModelResolverTests(unittest.TestCase):
    def test_image_backend_queues_resolved_workflow(self):
        from feverslop.adapters.comfyui_rendering import ComfyUIImageBackend
        from feverslop.ports.rendering import ImageRenderRequest, WorkflowAnchorConfig

        class Client:
            def __init__(self):
                self.queued_workflow = None

            def queue_prompt(self, workflow):
                self.queued_workflow = workflow
                return "prompt-id"

            def wait_for_completion(self, prompt_id):
                return {"outputs": {"save": {"images": [{"filename": "scene.png"}]}}}

            def extract_output_images(self, history):
                return history["outputs"]["save"]["images"]

            def download_view_file(self, *, filename, subfolder, file_type, output_path):
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                Path(output_path).write_bytes(b"png")
                return Path(output_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            workflow_path = temp / "image.json"
            workflow_path.write_text(
                json.dumps(
                    {
                        "1": {"inputs": {"text": ""}, "_meta": {"title": "#PROMPT"}},
                        "2": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_IMAGE"}},
                    }
                ),
                encoding="utf-8",
            )
            client = Client()
            resolver = FakeModelResolver()

            backend = ComfyUIImageBackend(
                client=client,
                workflow_path=workflow_path,
                output_dir=temp,
                model_resolver=resolver,
            )

            backend.render_image(
                ImageRenderRequest(
                    scene={},
                    scene_number=1,
                    prompt="prompt",
                    workflow_path=workflow_path,
                    output_dir=temp,
                    anchors=WorkflowAnchorConfig(
                        positive_prompt_title="#PROMPT",
                        negative_prompt_title=None,
                        save_image_title="#SAVE_IMAGE",
                        character_lora_title=None,
                    ),
                )
            )

            self.assertIn("resolved", client.queued_workflow)
            self.assertEqual(workflow_path, resolver.calls[0][1])

    def test_image_backend_does_not_patch_lora_strength_when_unset(self):
        from feverslop.adapters.comfyui_rendering import ComfyUIImageBackend
        from feverslop.ports.rendering import ImageRenderRequest, WorkflowAnchorConfig

        class Client:
            def __init__(self):
                self.queued_workflow = None

            def queue_prompt(self, workflow):
                self.queued_workflow = workflow
                return "prompt-id"

            def wait_for_completion(self, prompt_id):
                return {"outputs": {"save": {"images": [{"filename": "scene.png"}]}}}

            def extract_output_images(self, history):
                return history["outputs"]["save"]["images"]

            def download_view_file(self, *, filename, subfolder, file_type, output_path):
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                Path(output_path).write_bytes(b"png")
                return Path(output_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            workflow_path = temp / "image.json"
            workflow_path.write_text(
                json.dumps(
                    {
                        "1": {"inputs": {"text": ""}, "_meta": {"title": "#PROMPT"}},
                        "2": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_IMAGE"}},
                        "3": {
                            "inputs": {"lora_name": "workflow-default.safetensors", "strength_model": 0.42},
                            "_meta": {"title": "#LORA_1"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            client = Client()
            backend = ComfyUIImageBackend(
                client=client,
                workflow_path=workflow_path,
                output_dir=temp,
            )

            backend.render_image(
                ImageRenderRequest(
                    scene={},
                    scene_number=1,
                    prompt="prompt",
                    workflow_path=workflow_path,
                    output_dir=temp,
                    anchors=WorkflowAnchorConfig(
                        positive_prompt_title="#PROMPT",
                        negative_prompt_title=None,
                        save_image_title="#SAVE_IMAGE",
                        character_lora_title="#LORA_1",
                    ),
                )
            )

            self.assertEqual(0.42, client.queued_workflow["3"]["inputs"]["strength_model"])


class ComfyUIVideoBackendOrchestrationTests(unittest.TestCase):
    def _render_plan_scene(self) -> dict:
        return {
            "scene": 1,
            "abs_start_seconds": 0.0,
            "duration_seconds": 2.0,
            "frame_count": 48,
            "fps": 24,
            "width": 1280,
            "height": 704,
            "ltx": {"original_style_i2v_prompt": "prompt"},
        }

    def test_video_backend_module_does_not_import_workflow_patcher_directly(self):
        text = Path("src/feverslop/adapters/comfyui_video_backend.py").read_text(encoding="utf-8")

        self.assertNotIn("from feverslop.adapters.workflow_patcher import WorkflowPatcher", text)

    def test_video_backend_exposes_injected_collaborators(self):
        from feverslop.adapters.comfyui_video_backend import ComfyUIVideoBackendConfig, ComfyUIVideoRenderBackend

        backend = ComfyUIVideoRenderBackend(
            client=object(),
            config=ComfyUIVideoBackendConfig(ltx_workflow_path=Path("workflow.json"), output_dir=Path("out")),
            asset_uploader=FakeAssetUploader(),
            workflow_patcher=FakeWorkflowPatcher(),
            render_queue=FakeRenderQueue(),
            postprocessor=FakePostprocessor(),
        )

        self.assertIsInstance(backend.asset_uploader, FakeAssetUploader)
        self.assertIsInstance(backend.workflow_patcher, FakeWorkflowPatcher)
        self.assertIsInstance(backend.render_queue, FakeRenderQueue)
        self.assertIsInstance(backend.postprocessor, FakePostprocessor)

    def test_video_backend_uses_render_output_writer_collaborator(self):
        from feverslop.adapters.comfyui_video_backend import ComfyUIVideoRenderBackend

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            render_plan = temp / "render_plan.json"
            render_plan.write_text(json.dumps([self._render_plan_scene()]), encoding="utf-8")
            storyboard_dir = temp / "storyboard"
            storyboard_dir.mkdir()
            (storyboard_dir / "scene_0001.png").write_bytes(b"png")
            postprocessor = FakePostprocessor()
            backend = ComfyUIVideoRenderBackend(
                client=object(),
                ltx_workflow_path=temp / "workflow.json",
                output_dir=temp / "ltx",
                asset_uploader=FakeAssetUploader(),
                workflow_patcher=FakeWorkflowPatcher(),
                render_queue=FakeRenderQueue(),
                postprocessor=postprocessor,
            )

            backend.render_videos(render_plan, temp / "song.mp3", storyboard_dir)

            self.assertIs(backend.output_writer.postprocessor, postprocessor)
            self.assertEqual(temp / "ltx" / "render_manifest.json", postprocessor.manifest_calls[0][1])

    def test_video_backend_queues_resolved_workflow_after_dynamic_patching(self):
        from feverslop.adapters.comfyui_video_backend import ComfyUIVideoRenderBackend
        from feverslop.domain.ltx_rendering import AudioWindowSpec

        with tempfile.TemporaryDirectory() as temp_dir:
            workflow_patcher = FakeWorkflowPatcher()
            render_queue = FakeRenderQueue()
            resolver = FakeModelResolver()
            backend = ComfyUIVideoRenderBackend(
                client=object(),
                ltx_workflow_path="relay.json",
                output_dir=Path(temp_dir) / "out",
                workflow_patcher=workflow_patcher,
                render_queue=render_queue,
                model_resolver=resolver,
            )

            backend.render_scene_video(
                scene=self._render_plan_scene(),
                comfy_audio_name="audio.mp3",
                comfy_startframe_name="scene.png",
                rolling=AudioWindowSpec(
                    scene_frame_count=48,
                    render_frame_count=48,
                    trim_front_frames=0,
                    tail_loss_frames=0,
                    fps=24,
                    audio_start_seconds=0,
                    audio_duration_seconds=2,
                ),
            )

            queued_workflow = render_queue.calls[0][0]
            self.assertEqual({"scene": 1}, resolver.calls[0][0])
            self.assertEqual(Path("single_prompt.json"), resolver.calls[0][1])
            self.assertIn("resolved", queued_workflow)

    def test_render_videos_runs_with_fake_collaborators_without_comfyui_client(self):
        from feverslop.adapters.comfyui_video_backend import ComfyUIVideoRenderBackend

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            render_plan = temp / "render_plan.json"
            render_plan.write_text(json.dumps([self._render_plan_scene()]), encoding="utf-8")
            storyboard_dir = temp / "storyboard"
            storyboard_dir.mkdir()
            (storyboard_dir / "scene_0001.png").write_bytes(b"png")
            asset_uploader = FakeAssetUploader()
            workflow_patcher = FakeWorkflowPatcher()
            render_queue = FakeRenderQueue()
            postprocessor = FakePostprocessor()
            backend = ComfyUIVideoRenderBackend(
                client=object(),
                ltx_workflow_path=temp / "workflow.json",
                output_dir=temp / "ltx",
                asset_uploader=asset_uploader,
                workflow_patcher=workflow_patcher,
                render_queue=render_queue,
                postprocessor=postprocessor,
            )

            rendered = backend.render_videos(
                render_plan_path=render_plan,
                audio_file=temp / "song.mp3",
                storyboard_dir=storyboard_dir,
            )

            self.assertEqual([temp / "ltx" / "final" / "scene_0001.mp4"], rendered)
            self.assertEqual([(temp / "song.mp3", True, None)], asset_uploader.audio_calls)
            self.assertEqual([(storyboard_dir / "scene_0001.png", True)], asset_uploader.startframe_calls)
            self.assertEqual([(1, "audio/comfy-song.mp3", "storyboard/comfy-scene.png", 48)], workflow_patcher.workflow_calls)
            self.assertEqual([({"scene": 1}, 1, temp / "ltx" / "raw" / "scene_0001_raw.mp4")], render_queue.calls)
            self.assertEqual(1, len(postprocessor.trim_specs))
            self.assertEqual([([temp / "ltx" / "final" / "scene_0001.mp4"], temp / "ltx" / "concat_list.txt")], postprocessor.concat_calls)
            self.assertEqual(1, len(postprocessor.manifest_calls[0][0]))
            self.assertEqual(temp / "ltx" / "render_manifest.json", postprocessor.manifest_calls[0][1])

    def test_render_scene_video_builds_real_workflow_and_delegates_to_queue(self):
        from feverslop.adapters.comfyui_video_backend import ComfyUIVideoRenderBackend
        from feverslop.domain.ltx_rendering import AudioWindowSpec

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            workflow_path = temp / "single.json"
            workflow_path.write_text(
                json.dumps({
                    "1": {"inputs": {"value": 0}, "_meta": {"title": "#WIDTH"}},
                    "2": {"inputs": {"value": 0}, "_meta": {"title": "#HEIGHT"}},
                    "3": {"inputs": {"audio": "", "audioUI": ""}, "_meta": {"title": "#LOAD_AUDIO"}},
                    "4": {"inputs": {"start_index": 0, "duration": 0}, "_meta": {"title": "#TRIM_AUDIO"}},
                    "5": {"inputs": {"image": ""}, "_meta": {"title": "#STARTFRAME"}},
                    "6": {"inputs": {"value": 0}, "_meta": {"title": "#FRAMES"}},
                    "7": {"inputs": {"value": 0}, "_meta": {"title": "#FRAMERATE"}},
                    "8": {"inputs": {"noise_seed": 0}, "_meta": {"title": "#SEED"}},
                    "9": {"inputs": {"text": ""}, "_meta": {"title": "#PROMPT"}},
                    "10": {"inputs": {"filename_prefix": ""}, "_meta": {"title": "#SAVE_VIDEO"}},
                }),
                encoding="utf-8",
            )

            class CapturingQueue:
                def __init__(self):
                    self.calls = []

                def queue_workflow_and_download_first_video(self, workflow, *, scene_number, output_path):
                    self.calls.append((workflow, scene_number, Path(output_path)))
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(output_path).write_bytes(b"raw")
                    return Path(output_path)

            queue = CapturingQueue()
            backend = ComfyUIVideoRenderBackend(
                client=object(),
                ltx_workflow_path=workflow_path,
                output_dir=temp / "ltx",
                render_mode="single_prompt",
                render_queue=queue,
            )

            output = backend.render_scene_video(
                scene={
                    "scene": 1,
                    "fps": 24,
                    "width": 1280,
                    "height": 704,
                    "ltx": {"original_style_i2v_prompt": "prompt"},
                },
                comfy_audio_name="audio.mp3",
                comfy_startframe_name="scene_0001.png",
                rolling=AudioWindowSpec(
                    scene_frame_count=24,
                    render_frame_count=24,
                    trim_front_frames=0,
                    tail_loss_frames=0,
                    fps=24,
                    audio_start_seconds=0,
                    audio_duration_seconds=1,
                ),
            )

            workflow, scene_number, raw_path = queue.calls[0]
            self.assertEqual(temp / "ltx" / "raw" / "scene_0001_raw.mp4", output)
            self.assertEqual(1, scene_number)
            self.assertEqual(temp / "ltx" / "raw" / "scene_0001_raw.mp4", raw_path)
            self.assertEqual("prompt", workflow["9"]["inputs"]["text"])
            self.assertEqual("audio.mp3", workflow["3"]["inputs"]["audio"])
            self.assertEqual("scene_0001.png", workflow["5"]["inputs"]["image"])


if __name__ == "__main__":
    unittest.main()
