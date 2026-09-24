"""Tests for the H3-native FaceRefine backend (issue 519, unit 2).

Covers the ``ComfyUIFaceFixH3Backend`` adapter (workflow patching, preflight
node-class checks, video-to-video render), the H3 16k+5 frame-grid helpers,
and the ``_run_h3_facefix`` composition wiring (skip-existing, missing-source
skip, and the not-overwrite-source fallback).
"""
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from feverslop.adapters.comfyui_facefix_h3_backend import ComfyUIFaceFixH3Backend
from feverslop.domain.facefix_rendering import (
    is_h3_frame_grid_compatible,
    snap_to_h3_frame_grid,
)

WORKFLOW = "workflows/video_minimax_h3_facefix_v1.json"


class TestH3FrameGrid(unittest.TestCase):
    def test_grid_values_are_compatible(self):
        for n in (5, 21, 37, 197, 213):
            self.assertTrue(is_h3_frame_grid_compatible(n))

    def test_non_grid_values_are_incompatible(self):
        for n in (0, 20, 22, 200, 100):
            self.assertFalse(is_h3_frame_grid_compatible(n))

    def test_snap_rounds_to_nearest_grid_value(self):
        self.assertEqual(snap_to_h3_frame_grid(197), 197)
        self.assertEqual(snap_to_h3_frame_grid(200), 197)
        self.assertEqual(snap_to_h3_frame_grid(210), 213)
        self.assertEqual(snap_to_h3_frame_grid(190), 197)

    def test_snap_clamps_nonpositive_to_offset(self):
        self.assertEqual(snap_to_h3_frame_grid(0), 5)
        self.assertEqual(snap_to_h3_frame_grid(-3), 5)


def _make_backend(**kwargs):
    mock_client = MagicMock()
    mock_queue = MagicMock()
    mock_queue.queue_workflow_and_download_first_video.return_value = Path("/tmp/raw.mp4")
    mock_client.upload_file_via_image_endpoint.return_value = {
        "name": "scene.mp4", "subfolder": "", "type": "input",
    }
    mock_client.upload_image.return_value = {
        "name": "face.png", "subfolder": "", "type": "input",
    }
    return ComfyUIFaceFixH3Backend(
        client=mock_client,
        workflow_path=Path(WORKFLOW),
        render_queue=mock_queue,
        **kwargs,
    )


class TestH3FaceFixBackendBuildWorkflow(unittest.TestCase):
    def test_build_patches_source_prompt_ref_frame_denoise_save(self):
        backend = _make_backend()
        wf = backend.build_workflow(
            1,
            source_video=Path("/tmp/scene_0001.mp4"),
            actor_id="actor_a",
            face_ref_image=Path("/tmp/ref.png"),
            scene_prompt="Refine the face only.",
            frame_count=197,
            denoise=0.3,
        )
        # source video patched
        load = next(n for n in wf.values() if n.get("_meta", {}).get("title") == "#LOAD_SOURCE")
        self.assertEqual(load["inputs"]["video"], "scene.mp4")
        # prompt patched
        prompt = next(n for n in wf.values() if n.get("_meta", {}).get("title") == "#PROMPT")
        self.assertEqual(prompt["inputs"]["value"], "Refine the face only.")
        # frame count on grid (197 stays 197)
        fc = next(n for n in wf.values() if n.get("_meta", {}).get("title") == "#FRAMECOUNT")
        self.assertEqual(fc["inputs"]["value"], 197)
        # denoise patched
        sched = next(n for n in wf.values() if n.get("_meta", {}).get("title") == "#SCHEDULER")
        self.assertEqual(sched["inputs"]["denoise"], 0.3)
        # save prefix patched
        save = next(n for n in wf.values() if n.get("_meta", {}).get("title") == "#SAVE_VIDEO")
        self.assertEqual(save["inputs"]["filename_prefix"], "feverslop/facefix_h3/scene_0001_actor_a")

    def test_build_snapshots_off_grid_frame_count(self):
        backend = _make_backend()
        wf = backend.build_workflow(
            2,
            source_video=Path("/tmp/scene_0002.mp4"),
            actor_id="actor_b",
            frame_count=200,  # off grid -> 197
        )
        fc = next(n for n in wf.values() if n.get("_meta", {}).get("title") == "#FRAMECOUNT")
        self.assertEqual(fc["inputs"]["value"], 197)

    def test_build_without_ref_or_prompt_uses_defaults(self):
        backend = _make_backend()
        wf = backend.build_workflow(
            3,
            source_video=Path("/tmp/scene_0003.mp4"),
            actor_id="actor_c",
        )
        # No exception; ref/prompt/framecount left at workflow defaults.
        self.assertIn("137", wf)


class TestH3FaceFixBackendPreflight(unittest.TestCase):
    def test_missing_required_node_raises(self):
        backend = _make_backend()
        backend.client.get_object_info.return_value = ["VAELoader", "UNETLoader"]
        wf = backend.load_workflow()
        with self.assertRaises(RuntimeError) as ctx:
            backend._preflight_comfy_node_classes(wf)
        self.assertIn("missing required", str(ctx.exception))

    def test_missing_optional_inject_node_only_warns(self):
        backend = _make_backend()
        # All nodes present except the optional H3InjectVideoLatent.
        present = {
            "ResolutionSelector", "VAELoader", "VAEDecode", "KSamplerSelect",
            "BasicScheduler", "SamplerCustomAdvanced", "BasicGuider", "UNETLoader",
            "CLIPLoader", "RandomNoise", "PrimitiveInt", "MiniMaxH3ReferenceToVideo",
            "LoadImage", "PrimitiveStringMultiline", "VRAMCleanup", "VHS_VideoCombine",
            "LoraLoaderModelOnly", "VHS_LoadVideo",
        }
        backend.client.get_object_info.return_value = sorted(present)
        # Should not raise (H3InjectVideoLatent is optional).
        backend._preflight_comfy_node_classes(backend.load_workflow())

    def test_no_object_info_method_is_noop(self):
        backend = _make_backend()
        backend.client.get_object_info = None
        # Should not raise.
        backend._preflight_comfy_node_classes(backend.load_workflow())


class TestH3FaceFixBackendRenderScene(unittest.TestCase):
    def test_render_writes_workflow_and_returns_refined(self):
        backend = _make_backend()
        backend.client.get_object_info.return_value = [
            "ResolutionSelector", "VAELoader", "VAEDecode", "KSamplerSelect",
            "BasicScheduler", "SamplerCustomAdvanced", "BasicGuider", "UNETLoader",
            "CLIPLoader", "RandomNoise", "PrimitiveInt", "MiniMaxH3ReferenceToVideo",
            "LoadImage", "PrimitiveStringMultiline", "VRAMCleanup", "VHS_VideoCombine",
            "LoraLoaderModelOnly", "VHS_LoadVideo", "H3InjectVideoLatent",
        ]

        def _queue(workflow, *, scene_number, output_path):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"raw")
            return output_path

        backend.render_queue.queue_workflow_and_download_first_video.side_effect = _queue
        backend.render_scene(
            1,
            source_video=Path("/tmp/scene_0001.mp4"),
            output_dir=Path("/tmp/out"),
            actor_id="actor_a",
            frame_count=197,
            denoise=0.3,
        )
        backend.render_queue.queue_workflow_and_download_first_video.assert_called_once()
        # postprocess=True -> copies raw to refined_<actor>.mp4
        self.assertTrue(Path("/tmp/out/refined_actor_a.mp4").exists())
        # workflow artifact written
        self.assertTrue(Path("/tmp/out/workflow_facefix_h3.json").exists())


class TestRunH3FaceFixWiring(unittest.TestCase):
    def _patch_runtime(self):
        from feverslop.composition import facefix_pipeline as fp
        app_config = MagicMock()
        app_config.comfyui.ffmpeg_timeout_seconds = 120.0
        return [
            patch.object(fp, "_facefix_runtime", return_value=(MagicMock(), MagicMock(), MagicMock(), app_config)),
            patch.object(fp, "ComfyUIFaceFixH3Backend"),
            patch.object(fp, "InsightFaceDetectorAdapter"),
            patch.object(fp, "FaceIdentityAdapter"),
            patch.object(fp, "FaceMaskAdapter"),
            patch.object(fp, "coerce_local_path", side_effect=lambda p: Path(p) if p else Path(WORKFLOW)),
        ]

    def test_skip_existing_returns_existing_artifact(self):
        from feverslop.composition.facefix_pipeline import (
            FaceFixCompositionOptions,
            _run_h3_facefix,
        )
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = Path(tmp) / "scene_0001"
            scene_dir.mkdir()
            (scene_dir / "final.mp4").write_bytes(b"valid")
            (scene_dir / "final_facefix.mp4").write_bytes(b"valid")
            options = FaceFixCompositionOptions(
                scenes_dir=tmp,
                video_pipeline="minimax-h3-r2v",
                scene_numbers=[1],
                skip_existing=True,
                max_skip_rate=1.0,
            )
            patches = self._patch_runtime()
            for p in patches:
                p.start()
            try:
                result = _run_h3_facefix(options, console=None)
            finally:
                for p in patches:
                    p.stop()
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].name, "final_facefix.mp4")

    def test_missing_source_is_skipped(self):
        from feverslop.composition.facefix_pipeline import (
            FaceFixCompositionOptions,
            _run_h3_facefix,
        )
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "scene_0001").mkdir()
            options = FaceFixCompositionOptions(
                scenes_dir=tmp,
                video_pipeline="minimax-h3-r2v",
                scene_numbers=[1],
                skip_existing=False,
                max_skip_rate=1.0,
            )
            patches = self._patch_runtime()
            for p in patches:
                p.start()
            try:
                result = _run_h3_facefix(options, console=None)
            finally:
                for p in patches:
                    p.stop()
            self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
