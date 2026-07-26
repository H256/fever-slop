import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from feverslop.adapters.comfyui_facefix_backend import ComfyUIFaceFixRenderBackend
from feverslop.domain.facefix_rendering import FaceFixConfig, FaceFixSceneRequest


class TestFaceFixBackendBuildWorkflow(unittest.TestCase):
    def test_backend_builds_workflow_with_video_and_faces(self):
        mock_client = MagicMock()
        mock_queue = MagicMock()
        mock_queue.queue_workflow_and_download_first_video.return_value = Path("/tmp/raw.mp4")
        mock_client.upload_file_via_image_endpoint.return_value = {"name": "scene_0001.mp4", "subfolder": "", "type": "input"}
        mock_client.upload_image.return_value = {"name": "face.png", "subfolder": "feverslop/facefix/references/scene_0001", "type": "input"}

        cfg = FaceFixConfig()
        with patch.object(ComfyUIFaceFixRenderBackend, "load_workflow") as mock_load:
            mock_load.return_value = {
                "1": {
                    "class_type": "VHS_LoadVideo",
                    "_meta": {"title": "#LOAD_VIDEO"},
                    "inputs": {"video": ""},
                },
                "2": {
                    "class_type": "LoadImagesFromFolderKJ",
                    "_meta": {"title": "#FACE_REFS"},
                    "inputs": {"folder": ""},
                },
                "3": {
                    "class_type": "LTXVLoopingSampler",
                    "_meta": {"title": "#LOOPING_SAMPLER"},
                    "inputs": {
                        "guiding_strength": 0.0,
                        "cond_image_strength": 0.0,
                        "optional_cond_image_indices": "",
                        "optional_cond_images": ["2", 0],
                    },
                },
                "4": {
                    "class_type": "SaveAnimatedWEBM",
                    "_meta": {"title": "#SAVE_VIDEO"},
                    "inputs": {"filename_prefix": ""},
                },
            }

            with patch("feverslop.adapters.comfyui_facefix_backend.ComfyUIVideoAssetUploader"):
                with patch("feverslop.adapters.comfyui_facefix_backend.VideoPostProcessor"):
                    backend = ComfyUIFaceFixRenderBackend(
                        client=mock_client,
                        workflow_path=Path("workflows/facefix.json"),
                        output_dir=Path("/tmp/out"),
                        config=cfg,
                        render_queue=mock_queue,
                    )

                    request = FaceFixSceneRequest(
                        scene_number=1,
                        source_video=Path("/tmp/scene_0001.mp4"),
                        reference_images=[Path("/tmp/face.png")],
                        output_dir=Path("/tmp/out"),
                    )
                    scene = {"scene": 1, "fps": 24, "frame_count": 49}
                    workflow = backend.build_workflow(scene, request=request)
                    self.assertIsInstance(workflow, dict)


class TestFaceFixBackendInit(unittest.TestCase):
    def test_init_with_defaults(self):
        mock_client = MagicMock()
        backend = ComfyUIFaceFixRenderBackend(
            client=mock_client,
            workflow_path=Path("workflows/facefix.json"),
            output_dir=Path("/tmp/out"),
        )
        self.assertEqual(str(backend.workflow_path), str(Path("workflows/facefix.json")))
        self.assertEqual(str(backend.output_dir), str(Path("/tmp/out")))
        self.assertIsInstance(backend.config, FaceFixConfig)

    def test_init_with_config(self):
        mock_client = MagicMock()
        cfg = FaceFixConfig(guiding_strength=0.5)
        backend = ComfyUIFaceFixRenderBackend(
            client=mock_client,
            workflow_path=Path("workflows/facefix.json"),
            output_dir=Path("/tmp/out"),
            config=cfg,
        )
        self.assertEqual(backend.config.guiding_strength, 0.5)


class TestFaceFixBackendLoadWorkflow(unittest.TestCase):
    def test_load_workflow_from_path(self):
        mock_client = MagicMock()
        mock_workflow = {"1": {"class_type": "TestNode"}}
        with patch("pathlib.Path.read_text") as mock_read:
            mock_read.return_value = json.dumps(mock_workflow)
            backend = ComfyUIFaceFixRenderBackend(
                client=mock_client,
                workflow_path=Path("workflows/facefix.json"),
                output_dir=Path("/tmp/out"),
            )
            result = backend.load_workflow()
            self.assertEqual(result, mock_workflow)

    def test_load_workflow_from_memory(self):
        mock_client = MagicMock()
        mock_workflow = {"1": {"class_type": "TestNode"}}
        backend = ComfyUIFaceFixRenderBackend(
            client=mock_client,
            workflow_path=Path("workflows/facefix.json"),
            output_dir=Path("/tmp/out"),
            workflow=mock_workflow,
        )
        result = backend.load_workflow()
        self.assertEqual(result, mock_workflow)


if __name__ == "__main__":
    unittest.main()
