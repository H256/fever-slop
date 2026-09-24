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


class TestRunH3FaceFixSequentialChaining(unittest.TestCase):
    """U3: multi-actor scenes refine one subject at a time, chaining the
    composited output of one pass into the next actor's source."""

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

    def test_multi_actor_chains_composited_output(self):
        from types import SimpleNamespace
        from feverslop.composition.facefix_pipeline import (
            FaceFixCompositionOptions,
            _run_h3_facefix,
        )
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = Path(tmp) / "scene_0001"
            scene_dir.mkdir()
            (scene_dir / "final.mp4").write_bytes(b"valid")
            options = FaceFixCompositionOptions(
                scenes_dir=tmp,
                video_pipeline="minimax-h3-r2v",
                scene_numbers=[1],
                skip_existing=False,
                max_skip_rate=1.0,
                reference_images=[
                    Path(tmp) / "actors" / "actor_a" / "ref" / "sheet.png",
                    Path(tmp) / "actors" / "actor_b" / "ref" / "sheet.png",
                ],
            )
            patches = self._patch_runtime()
            # Two registered actor identities.
            extractor = MagicMock()
            extractor.extract_face_from_image.return_value = object()
            extractor.extract_embedding.return_value = object()
            # Two frames, one per actor (deterministic identity routing).
            fr_a = SimpleNamespace(processed=True, identity_actor_id="actor_a")
            fr_b = SimpleNamespace(processed=True, identity_actor_id="actor_b")
            pipeline_mock = MagicMock()
            pipeline_mock.process_frame.side_effect = [fr_a, fr_b]
            # _process_actor_h3 records the source it is called with and
            # returns a usable repair for every actor.
            process_mock = MagicMock(return_value=MagicMock())
            save_mock = MagicMock()
            compositor_cls = MagicMock()
            compositor_cls.return_value.composite.return_value = SimpleNamespace(
                composited_frames=[0, 1], diagnostic_mask_path=None
            )
            patches += [
                patch("feverslop.adapters.insightface_extractor.InsightFaceExtractor", return_value=extractor),
                patch.object(__import__("feverslop.composition.facefix_pipeline", fromlist=["x"]), "FacePipeline", return_value=pipeline_mock),
                patch.object(__import__("feverslop.composition.facefix_pipeline", fromlist=["x"]), "_process_actor_h3", process_mock),
                patch.object(__import__("feverslop.composition.facefix_pipeline", fromlist=["x"]), "_save_video_frames", save_mock),
                patch.object(__import__("feverslop.composition.facefix_pipeline", fromlist=["x"]), "_load_video_frames", return_value=[0, 1]),
                patch.object(__import__("feverslop.composition.facefix_pipeline", fromlist=["x"]), "FaceCompositor", compositor_cls),
            ]
            for p in patches:
                p.start()
            try:
                result = _run_h3_facefix(options, console=None)
            finally:
                for p in patches:
                    p.stop()

            # Two actors -> two sequential passes.
            self.assertEqual(process_mock.call_count, 2)
            # Deterministic sorted order: actor_a first, then actor_b.
            first_source = process_mock.call_args_list[0].kwargs["source"]
            second_source = process_mock.call_args_list[1].kwargs["source"]
            # First pass uses the original source render.
            self.assertEqual(first_source.name, "final.mp4")
            # Second pass chains the composited output of the first pass.
            self.assertEqual(second_source.name, "sequential_actor_b.mp4")
            # The composited chain is written back to the final artifact.
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].name, "final_facefix.mp4")

    def test_no_first_actor_fallback_for_multi_actor(self):
        from types import SimpleNamespace
        from feverslop.composition.facefix_pipeline import (
            FaceFixCompositionOptions,
            _run_h3_facefix,
        )
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = Path(tmp) / "scene_0001"
            scene_dir.mkdir()
            (scene_dir / "final.mp4").write_bytes(b"valid")
            options = FaceFixCompositionOptions(
                scenes_dir=tmp,
                video_pipeline="minimax-h3-r2v",
                scene_numbers=[1],
                skip_existing=False,
                max_skip_rate=1.0,
                reference_images=[
                    Path(tmp) / "actors" / "actor_a" / "ref" / "sheet.png",
                    Path(tmp) / "actors" / "actor_b" / "ref" / "sheet.png",
                ],
            )
            patches = self._patch_runtime()
            extractor = MagicMock()
            extractor.extract_face_from_image.return_value = object()
            extractor.extract_embedding.return_value = object()
            # One frame whose identity does not match any registered actor ->
            # must NOT be silently assigned to the first actor.
            fr_unknown = SimpleNamespace(processed=True, identity_actor_id="actor_z")
            pipeline_mock = MagicMock()
            pipeline_mock.process_frame.return_value = fr_unknown
            process_mock = MagicMock(return_value=MagicMock())
            patches += [
                patch("feverslop.adapters.insightface_extractor.InsightFaceExtractor", return_value=extractor),
                patch.object(__import__("feverslop.composition.facefix_pipeline", fromlist=["x"]), "FacePipeline", return_value=pipeline_mock),
                patch.object(__import__("feverslop.composition.facefix_pipeline", fromlist=["x"]), "_process_actor_h3", process_mock),
                patch.object(__import__("feverslop.composition.facefix_pipeline", fromlist=["x"]), "_save_video_frames", MagicMock()),
                patch.object(__import__("feverslop.composition.facefix_pipeline", fromlist=["x"]), "_load_video_frames", return_value=[0, 1]),
                patch.object(__import__("feverslop.composition.facefix_pipeline", fromlist=["x"]), "FaceCompositor", MagicMock()),
            ]
            for p in patches:
                p.start()
            try:
                _run_h3_facefix(options, console=None)
            finally:
                for p in patches:
                    p.stop()

            # The unmatched frame is routed to "unknown", not the first actor.
            self.assertEqual(process_mock.call_count, 1)
            self.assertEqual(process_mock.call_args_list[0].kwargs["actor_id"], "unknown")

    def test_vram_boundary_cleanup_and_stage_marker(self):
        from feverslop.composition.facefix_pipeline import (
            FaceFixCompositionOptions,
            _run_h3_facefix,
        )
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = Path(tmp) / "scene_0001"
            scene_dir.mkdir()
            (scene_dir / "final.mp4").write_bytes(b"valid")
            options = FaceFixCompositionOptions(
                scenes_dir=tmp,
                video_pipeline="minimax-h3-r2v",
                scene_numbers=[1],
                skip_existing=False,
                max_skip_rate=1.0,
            )
            patches = self._patch_runtime()
            runtime_patch = patches[0]
            for p in patches:
                p.start()
            client = runtime_patch.__enter__().return_value[0]
            try:
                _run_h3_facefix(options, console=None)
            finally:
                for p in patches:
                    p.stop()
            # The ComfyUI client's cache/VRAM is cleared at BOTH boundaries:
            # before the FaceRefine pass and before the downstream upscale.
            self.assertGreaterEqual(client.free_cache_and_vram.call_count, 2)


if __name__ == "__main__":
    unittest.main()
