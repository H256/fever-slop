import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


class TestFaceFixCropPipeline(unittest.TestCase):
    @patch("feverslop.composition.facefix_pipeline._run_crop_facefix")
    @patch("feverslop.composition.facefix_pipeline._run_legacy_facefix")
    def test_use_crop_pipeline_true(self, mock_legacy, mock_crop):
        from feverslop.composition.facefix_pipeline import run_facefix, FaceFixCompositionOptions

        mock_crop.return_value = [Path("/tmp/result.mp4")]
        options = FaceFixCompositionOptions(
            scenes_dir="/tmp/scenes",
            project_dir="/tmp/project",
            use_crop_pipeline=True,
        )
        run_facefix(options)
        mock_crop.assert_called_once()
        mock_legacy.assert_not_called()

    @patch("feverslop.composition.facefix_pipeline._run_crop_facefix")
    @patch("feverslop.composition.facefix_pipeline._run_legacy_facefix")
    def test_use_crop_pipeline_false(self, mock_legacy, mock_crop):
        from feverslop.composition.facefix_pipeline import run_facefix, FaceFixCompositionOptions

        mock_legacy.return_value = [Path("/tmp/result.mp4")]
        options = FaceFixCompositionOptions(
            scenes_dir="/tmp/scenes",
            use_crop_pipeline=False,
        )
        run_facefix(options)
        mock_legacy.assert_called_once()
        mock_crop.assert_not_called()

    def test_default_use_crop_pipeline(self):
        from feverslop.composition.facefix_pipeline import FaceFixCompositionOptions
        options = FaceFixCompositionOptions()
        self.assertTrue(options.use_crop_pipeline)


class TestFaceFixCropBackendInit(unittest.TestCase):
    """Test ComfyUIFaceFixCropBackend __init__ with various parameter combos."""

    def test_init_defaults(self):
        from feverslop.adapters.comfyui_facefix_crop_backend import ComfyUIFaceFixCropBackend

        backend = ComfyUIFaceFixCropBackend(client=MagicMock())
        self.assertIsNotNone(backend.client)
        self.assertIsNotNone(backend.config)
        self.assertTrue(backend.postprocess)
        self.assertIsNone(backend.face_ref_image)

    def test_init_explicit_face_ref(self):
        from feverslop.adapters.comfyui_facefix_crop_backend import ComfyUIFaceFixCropBackend

        ref_path = Path("/tmp/ref.png")
        backend = ComfyUIFaceFixCropBackend(
            client=MagicMock(),
            face_ref_image=ref_path,
        )
        self.assertEqual(backend.face_ref_image, ref_path)

    def test_init_workflow_path(self):
        from feverslop.adapters.comfyui_facefix_crop_backend import ComfyUIFaceFixCropBackend

        wf_path = Path("/tmp/custom_workflow.json")
        backend = ComfyUIFaceFixCropBackend(
            client=MagicMock(),
            workflow_path=wf_path,
        )
        self.assertEqual(backend.workflow_path, wf_path)

    def test_init_postprocess_off(self):
        from feverslop.adapters.comfyui_facefix_crop_backend import ComfyUIFaceFixCropBackend

        backend = ComfyUIFaceFixCropBackend(
            client=MagicMock(),
            postprocess=False,
            postprocess_reencode=False,
        )
        self.assertFalse(backend.postprocess)


class TestFaceFixCropBackendBuildWorkflow(unittest.TestCase):
    """Test build_workflow with face_ref_image and anchor scenarios."""

    def _make_mock_workflow(self):
        return {
            "1": {"_meta": {"title": "#LOAD_VIDEO"}, "class_type": "LoadVideo", "inputs": {"video": "", "videopath": ""}},
            "2": {"_meta": {"title": "#LOOPING_SAMPLER"}, "class_type": "KSampler", "inputs": {"optional_cond_images": None}},
            "3": {"_meta": {"title": "#SAVE_VIDEO"}, "class_type": "SaveAnimatedWEBP", "inputs": {"filename_prefix": ""}},
        }

    def test_build_workflow_with_face_ref(self):
        from feverslop.adapters.comfyui_facefix_crop_backend import ComfyUIFaceFixCropBackend

        mock_client = MagicMock()
        mock_client.upload_file_via_image_endpoint.return_value = {"filename": "test.mp4", "subfolder": "test"}
        mock_client.upload_image.return_value = {"filename": "ref.png", "subfolder": "test"}

        backend = ComfyUIFaceFixCropBackend(
            client=mock_client,
            workflow_path=Path("workflows/test.json"),
        )

        with patch.object(backend, "load_workflow", return_value=self._make_mock_workflow()):
            workflow = backend.build_workflow(
                scene_number=1,
                face_crop_mp4=Path("/tmp/crop.mp4"),
                anchors_dir=Path("/tmp/anchors"),
                actor_id="hero",
                face_ref_image=Path("/tmp/face_ref.png"),
            )
            self.assertIsInstance(workflow, dict)

    def test_build_workflow_no_face_ref(self):
        from feverslop.adapters.comfyui_facefix_crop_backend import ComfyUIFaceFixCropBackend

        mock_client = MagicMock()
        mock_client.upload_file_via_image_endpoint.return_value = {"filename": "test.mp4", "subfolder": "test"}

        backend = ComfyUIFaceFixCropBackend(
            client=mock_client,
            workflow_path=Path("workflows/test.json"),
        )

        with patch.object(backend, "load_workflow", return_value=self._make_mock_workflow()):
            workflow = backend.build_workflow(
                scene_number=1,
                face_crop_mp4=Path("/tmp/crop.mp4"),
                anchors_dir=Path("/tmp/anchors"),
                actor_id="hero",
                face_ref_image=None,
            )
            self.assertIsInstance(workflow, dict)

    def test_build_workflow_face_ref_missing_file(self):
        from feverslop.adapters.comfyui_facefix_crop_backend import ComfyUIFaceFixCropBackend

        mock_client = MagicMock()
        mock_client.upload_file_via_image_endpoint.return_value = {"filename": "test.mp4", "subfolder": "test"}

        backend = ComfyUIFaceFixCropBackend(
            client=MagicMock(),
            face_ref_image=Path("/tmp/nonexistent.png"),
        )

        with patch.object(backend, "load_workflow", return_value=self._make_mock_workflow()):
            # Should not raise — missing file is silently skipped.
            workflow = backend.build_workflow(
                scene_number=1,
                face_crop_mp4=Path("/tmp/crop.mp4"),
                anchors_dir=Path("/tmp/anchors"),
                actor_id="hero",
            )
            self.assertIsInstance(workflow, dict)


class TestFaceVideoEncoder(unittest.TestCase):
    @patch("feverslop.adapters.face_video_encoder.subprocess")
    def test_encode_success(self, mock_subprocess):
        from feverslop.adapters.face_video_encoder import encode_face_crop_mp4

        mock_subprocess.run.return_value = MagicMock(returncode=0, stderr="")
        frames_dir = Path("/tmp/crops")
        output = Path("/tmp/face_crop.mp4")

        # Mock subprocess callback creates the output file so integrity check passes
        def _fake_run(*args, **kwargs):
            nonlocal output
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"fake_mp4_data")
            return mock_subprocess.run.return_value

        mock_subprocess.run.side_effect = _fake_run

        result = encode_face_crop_mp4(frames_dir, 24.0, output)
        self.assertEqual(result, output)
        mock_subprocess.run.assert_called_once()
        output.unlink(missing_ok=True)

    @patch("feverslop.adapters.face_video_encoder.subprocess")
    def test_encode_missing_output_raises(self, mock_subprocess):
        from feverslop.adapters.face_video_encoder import encode_face_crop_mp4

        mock_subprocess.run.return_value = MagicMock(returncode=0, stderr="")

        with self.assertRaises(RuntimeError) as ctx:
            encode_face_crop_mp4(Path("/tmp/crops"), 24.0, Path("/tmp/does_not_exist.mp4"))

        self.assertIn("no output file", str(ctx.exception))

    @patch("feverslop.adapters.face_video_encoder.subprocess")
    def test_encode_empty_output_raises(self, mock_subprocess):
        from feverslop.adapters.face_video_encoder import encode_face_crop_mp4

        mock_subprocess.run.return_value = MagicMock(returncode=0, stderr="")

        frames_dir = Path("/tmp/crops")
        output = Path("/tmp/empty_output.mp4")
        # Create empty file so it exists but is 0 bytes
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch()

        with self.assertRaises(RuntimeError) as ctx:
            encode_face_crop_mp4(frames_dir, 24.0, output)

        self.assertIn("empty", str(ctx.exception))

    @patch("feverslop.adapters.face_video_encoder.subprocess")
    def test_encode_failure_surfaces_stderr(self, mock_subprocess):
        from feverslop.adapters.face_video_encoder import encode_face_crop_mp4

        mock_subprocess.run.return_value = MagicMock(
            returncode=1, stderr="Invalid input format"
        )

        with self.assertRaises(RuntimeError) as ctx:
            encode_face_crop_mp4(Path("/tmp/crops"), 24.0, Path("/tmp/out.mp4"))

        self.assertIn("exit 1", str(ctx.exception))
        self.assertIn("Invalid input format", str(ctx.exception))


class TestFaceMaskAdapter(unittest.TestCase):
    """Test FaceMaskAdapter with MaskValidationResult."""

    def test_generate_mask_valid(self):
        from feverslop.adapters.face_mask import FaceMaskAdapter
        from feverslop.domain.face_detection import BoundingBox

        adapter = FaceMaskAdapter()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        box = BoundingBox(400, 300, 600, 500)

        mask = adapter.generate_mask(frame, box)
        self.assertEqual(mask.shape, (1080, 1920))
        self.assertTrue(np.any(mask > 0))

    def test_validate_mask_returns_result(self):
        from feverslop.adapters.face_mask import FaceMaskAdapter, MaskValidationResult

        adapter = FaceMaskAdapter()
        mask = np.ones((100, 100), dtype=np.uint8) * 255

        result = adapter.validate_mask(mask)
        self.assertIsInstance(result, MaskValidationResult)
        self.assertTrue(result.valid)

    def test_validate_mask_empty(self):
        from feverslop.adapters.face_mask import FaceMaskAdapter, MaskValidationResult

        adapter = FaceMaskAdapter()
        mask = np.zeros((0,), dtype=np.uint8)

        result = adapter.validate_mask(mask)
        self.assertIsInstance(result, MaskValidationResult)
        self.assertFalse(result.valid)
        self.assertIn("empty", result.message)

    def test_validate_mask_below_threshold(self):
        from feverslop.adapters.face_mask import FaceMaskAdapter

        adapter = FaceMaskAdapter()
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0, 0] = 1  # Single nonzero pixel — ratio = 0.0001

        result = adapter.validate_mask(mask, min_nonzero_ratio=0.01)
        self.assertFalse(result.valid)
        self.assertIn("nonzero_ratio", result.message)


class TestFaceDebugAdapter(unittest.TestCase):
    """Test that FaceDebugAdapter implements DebugArtifactPort."""

    def test_write_debug_image(self):
        from feverslop.adapters.face_debug import FaceDebugAdapter
        import tempfile
        import shutil

        tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        try:
            adapter = FaceDebugAdapter(tmpdir)
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            path = adapter.write_debug_image(0, image, "test")
            self.assertTrue(path.exists())
            self.assertIn("test_frame000000", str(path))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_write_detection_overlay(self):
        from feverslop.adapters.face_debug import FaceDebugAdapter
        from feverslop.domain.face_detection import FaceDetection, BoundingBox
        import tempfile
        import shutil

        tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        try:
            adapter = FaceDebugAdapter(tmpdir)
            frame = np.zeros((200, 300, 3), dtype=np.uint8)
            detections = [FaceDetection(
                box=BoundingBox(50, 50, 150, 150),
                score=0.95,
            )]
            path = adapter.write_detection_overlay(
                0, frame, detections, "ACCEPTED"
            )
            self.assertTrue(path.exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_write_crop(self):
        from feverslop.adapters.face_debug import FaceDebugAdapter
        import tempfile
        import shutil

        tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        try:
            adapter = FaceDebugAdapter(tmpdir)
            crop = np.zeros((768, 768, 3), dtype=np.uint8)
            path = adapter.write_crop(5, crop)
            self.assertTrue(path.exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_write_mask(self):
        from feverslop.adapters.face_debug import FaceDebugAdapter
        import tempfile
        import shutil

        tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        try:
            adapter = FaceDebugAdapter(tmpdir)
            mask = np.zeros((1080, 1920), dtype=np.uint8)
            path = adapter.write_mask(0, mask)
            self.assertTrue(path.exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_write_composite(self):
        from feverslop.adapters.face_debug import FaceDebugAdapter
        import tempfile
        import shutil

        tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        try:
            adapter = FaceDebugAdapter(tmpdir)
            original = np.zeros((100, 100, 3), dtype=np.uint8)
            processed = np.zeros((100, 100, 3), dtype=np.uint8)
            mask = np.zeros((100, 100), dtype=np.uint8)
            path = adapter.write_composite(0, original, processed, mask)
            self.assertTrue(path.exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestFaceIdentityAdapter(unittest.TestCase):
    """Test FaceIdentityAdapter basics."""

    def test_verify_identity_no_refs(self):
        from feverslop.adapters.face_identity import FaceIdentityAdapter

        adapter = FaceIdentityAdapter()
        emb = np.random.randn(512).astype(np.float32)
        actor_id, score = adapter.verify_identity(emb)
        self.assertIsNone(actor_id)
        self.assertIsNone(score)

    def test_register_and_verify(self):
        from feverslop.adapters.face_identity import FaceIdentityAdapter

        adapter = FaceIdentityAdapter(min_similarity=0.5)
        ref_emb = np.random.randn(512).astype(np.float32)
        adapter.register_reference(ref_emb, "hero")

        actor_id, score = adapter.verify_identity(ref_emb)
        self.assertEqual(actor_id, "hero")
        self.assertIsNotNone(score)

    def test_get_all_embeddings(self):
        from feverslop.adapters.face_identity import FaceIdentityAdapter

        adapter = FaceIdentityAdapter()
        ref1 = np.random.randn(512).astype(np.float32)
        ref2 = np.random.randn(512).astype(np.float32)
        adapter.register_reference(ref1, "hero")
        adapter.register_reference(ref2, "villain")

        embeddings = adapter.get_all_embeddings()
        self.assertEqual(len(embeddings), 2)
        actor_ids = {e.actor_id for e in embeddings}
        self.assertEqual(actor_ids, {"hero", "villain"})


class TestSceneArtifactLayoutExtensions(unittest.TestCase):
    def test_actor_methods(self):
        from feverslop.scene_artifacts import SceneArtifactLayout
        layout = SceneArtifactLayout(Path("/tmp/project"))

        self.assertEqual(layout.actors_dir_path(), Path("/tmp/project/output/references/actors"))

        scene_dir = layout.scene_facefix_dir(1, "hero")
        self.assertEqual(scene_dir, Path("/tmp/project/output/render/scenes/scene_0001/facefix/hero"))

        crop_mp4 = layout.scene_face_crop_mp4(1, "hero")
        self.assertEqual(crop_mp4, scene_dir / "face_crop.mp4")

        anchors = layout.scene_face_anchors_dir(1, "hero")
        self.assertEqual(anchors, scene_dir / "anchors")

        crops = layout.scene_face_crops_dir(1, "hero")
        self.assertEqual(crops, scene_dir / "crops")

        repaired = layout.scene_face_repaired_dir(1, "hero")
        self.assertEqual(repaired, scene_dir / "repaired")

    def test_actor_methods_extra(self):
        from feverslop.scene_artifacts import SceneArtifactLayout
        layout = SceneArtifactLayout(Path("/tmp/project"))

        facefix = layout.scene_final_facefix_video(1)
        self.assertEqual(facefix, Path("/tmp/project/output/render/scenes/scene_0001/final_facefix.mp4"))

        workflow = layout.scene_workflow_facefix(1)
        self.assertEqual(workflow, Path("/tmp/project/output/render/scenes/scene_0001/workflow_facefix.json"))

    def test_actor_ids_empty(self):
        from feverslop.scene_artifacts import SceneArtifactLayout
        layout = SceneArtifactLayout(Path("/tmp/nonexistent_project"))
        self.assertEqual(layout.actor_ids(), [])


class TestLoadVideoFrames(unittest.TestCase):
    @patch("feverslop.composition.facefix_pipeline.cv2")
    def test_load_success(self, mock_cv2):
        from feverslop.composition.facefix_pipeline import _load_video_frames

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, frame), (True, frame), (False, None)]
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.cvtColor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        result = _load_video_frames(Path("/tmp/video.mp4"))
        self.assertEqual(result.shape, (2, 480, 640, 3))

    @patch("feverslop.composition.facefix_pipeline.cv2")
    def test_load_fail(self, mock_cv2):
        from feverslop.composition.facefix_pipeline import _load_video_frames

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap

        result = _load_video_frames(Path("/tmp/missing.mp4"))
        self.assertIsNone(result)


class TestDeprecatedBackend(unittest.TestCase):
    """Verify ComfyUIFaceFixRenderBackend raises DeprecationWarning."""

    def test_deprecation_warning(self):
        with self.assertWarns(DeprecationWarning) as ctx:
            from feverslop.adapters.comfyui_facefix_backend import ComfyUIFaceFixRenderBackend
            ComfyUIFaceFixRenderBackend(
                client=MagicMock(),
                workflow_path=Path("/tmp/wf.json"),
            )
        self.assertIn("deprecated", str(ctx.warning).lower())


if __name__ == "__main__":
    unittest.main()
