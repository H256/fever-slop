import unittest
import json
import tempfile
from pathlib import Path


class FakeComfyUIClient:
    def __init__(self):
        self.audio_uploads = []
        self.image_uploads = []

    def upload_file_via_image_endpoint(self, path, *, subfolder, file_type, overwrite):
        self.audio_uploads.append((Path(path), subfolder, file_type, overwrite))
        return {"name": Path(path).name, "subfolder": subfolder}

    def upload_image(self, path, *, subfolder, file_type, overwrite):
        self.image_uploads.append((Path(path), subfolder, file_type, overwrite))
        return {"name": Path(path).name, "subfolder": subfolder}


class ComfyUIVideoAssetUploaderTests(unittest.TestCase):
    def test_upload_audio_uses_comfyui_image_endpoint_contract(self):
        from autoprompter.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader

        client = FakeComfyUIClient()
        uploader = ComfyUIVideoAssetUploader(client)

        name = uploader.resolve_audio_name(
            Path("song.mp3"),
            upload_audio=True,
            uploaded_audio_name=None,
        )

        self.assertEqual("autoprompter/audio/song.mp3", name)
        self.assertEqual(
            [(Path("song.mp3"), "autoprompter/audio", "input", True)],
            client.audio_uploads,
        )

    def test_audio_upload_can_be_skipped_with_uploaded_name_or_file_name(self):
        from autoprompter.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader

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
        from autoprompter.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader

        client = FakeComfyUIClient()
        uploader = ComfyUIVideoAssetUploader(client)

        name = uploader.resolve_startframe_name(
            Path("scene_0001.png"),
            upload_startframes=True,
        )

        self.assertEqual("autoprompter/storyboard/scene_0001.png", name)
        self.assertEqual(
            [(Path("scene_0001.png"), "autoprompter/storyboard", "input", True)],
            client.image_uploads,
        )

    def test_startframe_upload_can_be_skipped(self):
        from autoprompter.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader

        uploader = ComfyUIVideoAssetUploader(FakeComfyUIClient())

        self.assertEqual(
            "scene_0001.png",
            uploader.resolve_startframe_name(
                Path("scene_0001.png"),
                upload_startframes=False,
            ),
        )

    def test_malformed_upload_response_raises_clear_error(self):
        from autoprompter.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader

        with self.assertRaisesRegex(ValueError, "Unexpected ComfyUI upload response"):
            ComfyUIVideoAssetUploader.comfy_path_from_upload({"subfolder": "x"})


class LTXWorkflowPatcherTests(unittest.TestCase):
    def _workflow_with_titles(self, titles: list[str]) -> str:
        return "{" + ",".join(
            f'"{index}": {{"inputs": {{}}, "_meta": {{"title": "{title}"}}}}'
            for index, title in enumerate(titles, start=1)
        ) + "}"

    def _settings(self, temp: Path, **overrides):
        from autoprompter.adapters.ltx_workflow_patcher import LTXWorkflowSettings

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
            character_lora_node_title="#CHARACTER_LORA",
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
        from autoprompter.adapters.ltx_workflow_patcher import LTXWorkflowPatcher

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

            with self.assertRaisesRegex(ValueError, "#PROMPT_RELAY"):
                LTXWorkflowPatcher(settings).validate_workflow(mode="relay")

    def test_single_prompt_validation_accepts_prompt_positive_fallback(self):
        from autoprompter.adapters.ltx_workflow_patcher import LTXWorkflowPatcher

        with tempfile.TemporaryDirectory() as temp_dir:
            patcher = LTXWorkflowPatcher(self._settings(Path(temp_dir)))

            patcher.validate_workflow(mode="single_prompt")

    def test_lora_enabled_requires_lora_anchor(self):
        from autoprompter.adapters.ltx_workflow_patcher import LTXWorkflowPatcher

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

            with self.assertRaisesRegex(ValueError, "#LORA_1"):
                LTXWorkflowPatcher(settings).validate_workflow(mode="relay")

    def test_single_prompt_build_workflow_patches_original_style_prompt(self):
        from autoprompter.adapters.ltx_workflow_patcher import LTXWorkflowPatcher
        from autoprompter.domain.ltx_rendering import AudioWindowSpec

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
        from autoprompter.adapters.ltx_workflow_patcher import LTXWorkflowPatcher
        from autoprompter.domain.ltx_rendering import AudioWindowSpec

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
        from autoprompter.adapters.ltx_workflow_patcher import LTXWorkflowPatcher
        from autoprompter.domain.ltx_rendering import AudioWindowSpec

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


if __name__ == "__main__":
    unittest.main()
