"""Tests for ComfyUIMiniMaxH3R2VBackend."""

import unittest
import tempfile
import json
from pathlib import Path

from feverslop.adapters.comfyui_minimax_h3_r2v_backend import (
    ComfyUIMiniMaxH3R2VBackend,
)
from feverslop.adapters.workflow_patcher import WorkflowPatcher
from feverslop.domain.postprocessing import TrimSpec
from feverslop.errors import FeverSlopValidationError
from feverslop.ports.rendering import VideoRenderRequest


# ---------------------------------------------------------------------------
# Fake helpers
# ---------------------------------------------------------------------------

class FakeClient:
    pass


class FakeAssetUploader:
    def __init__(self):
        self.resolve_reference_image_calls: list[Path | str] = []
        self.resolve_audio_calls: list[tuple] = []

    def resolve_reference_image_name(self, image_path, **kwargs):
        self.resolve_reference_image_calls.append(image_path)
        p = Path(image_path)
        return f"feverslop/references/{p.stem}-abc123{p.suffix}"

    def resolve_audio_name(self, audio_file, *, upload_audio, uploaded_audio_name):
        self.resolve_audio_calls.append((audio_file, upload_audio, uploaded_audio_name))
        p = Path(audio_file)
        return f"feverslop/audio/{p.stem}-def456{p.suffix}"


class FakeRenderQueue:
    def __init__(self):
        self.calls: list[dict] = []

    def queue_workflow_and_download_first_video(self, workflow, *, scene_number, output_path):
        self.calls.append({
            "scene_number": scene_number,
            "output_path": output_path,
        })
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake video")
        return output_path


class FakePostProcessor:
    def __init__(self):
        self.trim_specs: list[TrimSpec] = []

    def trim_clip(self, spec: TrimSpec) -> Path:
        self.trim_specs.append(spec)
        spec.output_file.parent.mkdir(parents=True, exist_ok=True)
        spec.output_file.write_bytes(b"final")
        return spec.output_file


class FakeModelResolver:
    def resolve_workflow_models(self, workflow, workflow_path=None):
        return workflow


# ---------------------------------------------------------------------------
# Minimal R2V workflow fixture
# ---------------------------------------------------------------------------

def _native_r2v_workflow() -> dict:
    """Minimal R2V workflow with FeverSlop anchors (no audio)."""
    return {
        "10": {
            "class_type": "ResolutionSelector",
            "_meta": {"title": "#MEGAPIXELS"},
            "inputs": {"megapixels": 0.4, "aspect_ratio": "16:9", "multiple": 32},
        },
        "20": {
            "class_type": "RandomNoise",
            "_meta": {"title": "#SEED"},
            "inputs": {"noise_seed": 12345},
        },
        "30": {
            "class_type": "PrimitiveFloat",
            "_meta": {"title": "#DURATION"},
            "inputs": {"value": 5.0},
        },
        "40": {
            "class_type": "PrimitiveStringMultiline",
            "_meta": {"title": "#PROMPT"},
            "inputs": {"value": "old prompt"},
        },
        "50": {
            "class_type": "LoadImage",
            "_meta": {"title": "#REF_1"},
            "inputs": {"image": "placeholder.png"},
        },
        "60": {
            "class_type": "LoadImage",
            "_meta": {"title": "#REF_2"},
            "inputs": {"image": "placeholder2.png"},
        },
        "70": {
            "class_type": "VHS_VideoCombine",
            "_meta": {"title": "#SAVE_VIDEO"},
            "inputs": {"filename_prefix": "video/minimaxh3", "format": "video/h264-mp4"},
        },
        "80": {
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"},
            "inputs": {},
        },
        "81": {
            "class_type": "VAEDecodeAudio",
            "_meta": {"title": "VAE Decode Audio"},
            "inputs": {},
        },
    }


def _audio_r2v_workflow() -> dict:
    """R2V workflow with audio anchors (#LOAD_AUDIO + #TRIM_AUDIO)."""
    wf = _native_r2v_workflow()
    wf["90"] = {
        "class_type": "LoadAudio",
        "_meta": {"title": "#LOAD_AUDIO"},
        "inputs": {"audio": "old_audio.wav", "audioUI": "old_ui"},
    }
    wf["91"] = {
        "class_type": "TrimAudioDuration",
        "_meta": {"title": "#TRIM_AUDIO"},
        "inputs": {"start_index": 0, "duration": 5.0, "audio": ["90", 0]},
    }
    return wf


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------

class ConstructorTests(unittest.TestCase):
    def test_constructor_stores_params(self):
        uploader = FakeAssetUploader()
        backend = ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/workflow.json"),
            output_dir=Path("/tmp/output"),
            asset_uploader=uploader,
        )
        self.assertEqual(Path("/tmp/output"), backend.output_dir)
        self.assertEqual(Path("/tmp/output") / "raw", backend.raw_output_dir)
        self.assertIs(backend.asset_uploader, uploader)
        self.assertFalse(backend.randomize_seed)
        self.assertEqual(100000, backend.seed_offset)
        self.assertIsNone(backend.debug_workflows_dir)

    def test_constructor_seed_defaults(self):
        backend = ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
        )
        self.assertEqual(100000, backend.seed_offset)
        self.assertFalse(backend.randomize_seed)

    def test_constructor_custom_seed_settings(self):
        backend = ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
            seed_offset=50000,
            randomize_seed=True,
        )
        self.assertEqual(50000, backend.seed_offset)
        self.assertTrue(backend.randomize_seed)

    def test_constructor_debug_workflows_dir(self):
        backend = ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
            debug_workflows_dir=Path("/tmp/debug"),
        )
        self.assertEqual(Path("/tmp/debug"), backend.debug_workflows_dir)

    def test_constructor_in_memory_workflow(self):
        wf = _native_r2v_workflow()
        backend = ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
            workflow=wf,
        )
        loaded = backend.load_workflow()
        self.assertEqual(wf, loaded)
        self.assertIsNot(loaded, wf)

    def test_constructor_creates_defaults(self):
        backend = ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
        )
        self.assertIsNotNone(backend.asset_uploader)
        self.assertIsNotNone(backend.render_queue)
        self.assertIsNotNone(backend.postprocessor)
        self.assertIsNotNone(backend.model_resolver)

    def test_max_ref_images_constant(self):
        self.assertEqual(9, ComfyUIMiniMaxH3R2VBackend.MAX_REF_IMAGES)

    def test_fps_constant(self):
        self.assertEqual(24, ComfyUIMiniMaxH3R2VBackend.FPS)


# ---------------------------------------------------------------------------
# _validate_scene tests
# ---------------------------------------------------------------------------

class ValidateSceneTests(unittest.TestCase):
    def setUp(self):
        self.backend = ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
        )

    def test_no_refs_raises(self):
        with self.assertRaises(FeverSlopValidationError) as ctx:
            self.backend._validate_scene({"scene": 1})
        self.assertIn("actor reference", str(ctx.exception).lower())

    def test_empty_refs_raises(self):
        with self.assertRaises(FeverSlopValidationError):
            self.backend._validate_scene({"scene": 1, "references": {}})

    def test_one_actor_ok(self):
        scene = {
            "scene": 1,
            "references": {"actor_sheet_paths": ["/tmp/actor.png"]},
        }
        self.backend._validate_scene(scene)

    def test_five_actors_ok(self):
        scene = {
            "scene": 1,
            "references": {
                "actor_sheet_paths": [f"/tmp/a{i}.png" for i in range(5)],
            },
        }
        self.backend._validate_scene(scene)

    def test_actor_msr_paths_ok(self):
        scene = {
            "scene": 1,
            "references": {"actor_msr_paths": ["/tmp/actor.png"]},
        }
        self.backend._validate_scene(scene)


# ---------------------------------------------------------------------------
# _seed_for_scene tests
# ---------------------------------------------------------------------------

class SeedForSceneTests(unittest.TestCase):
    def test_sequential_seed(self):
        backend = ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
            seed_offset=100000,
            randomize_seed=False,
        )
        self.assertEqual(100001, backend._seed_for_scene(1))
        self.assertEqual(100005, backend._seed_for_scene(5))

    def test_randomized_seed(self):
        backend = ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
            randomize_seed=True,
        )
        seed_a = backend._seed_for_scene(1)
        seed_b = backend._seed_for_scene(1)
        # Two calls should produce different random seeds
        self.assertIsInstance(seed_a, int)
        self.assertGreaterEqual(seed_a, 0)
        self.assertNotEqual(seed_a, seed_b)


# ---------------------------------------------------------------------------
# _has_anchor tests
# ---------------------------------------------------------------------------

class HasAnchorTests(unittest.TestCase):
    def test_found(self):
        wf = _native_r2v_workflow()
        patcher = WorkflowPatcher(wf)
        self.assertTrue(
            ComfyUIMiniMaxH3R2VBackend._has_anchor(patcher, "#PROMPT")
        )

    def test_not_found(self):
        wf = _native_r2v_workflow()
        patcher = WorkflowPatcher(wf)
        self.assertFalse(
            ComfyUIMiniMaxH3R2VBackend._has_anchor(patcher, "#DOES_NOT_EXIST")
        )


# ---------------------------------------------------------------------------
# _patch_megapixels tests
# ---------------------------------------------------------------------------

class PatchMegapixelsTests(unittest.TestCase):
    def test_patches_megapixels_value(self):
        wf = _native_r2v_workflow()
        patcher = WorkflowPatcher(wf)
        ComfyUIMiniMaxH3R2VBackend._patch_megapixels(patcher, 0.5)
        self.assertEqual(0.5, patcher.get()["10"]["inputs"]["megapixels"])

    def test_rounds_to_one_decimal(self):
        wf = _native_r2v_workflow()
        patcher = WorkflowPatcher(wf)
        ComfyUIMiniMaxH3R2VBackend._patch_megapixels(patcher, 0.786432)
        self.assertEqual(0.8, patcher.get()["10"]["inputs"]["megapixels"])

    def test_computation_from_resolution(self):
        """Verify the formula: (width * height) / 1_000_000, rounded to 0.1."""
        mf = round(1344 * 768 / 1_000_000, 1)
        self.assertEqual(1.0, mf)
        mf2 = round(1024 * 768 / 1_000_000, 1)
        self.assertEqual(0.8, mf2)


# ---------------------------------------------------------------------------
# _patch_reference_images tests
# ---------------------------------------------------------------------------

class PatchReferenceImagesTests(unittest.TestCase):
    def _backend(self):
        uploader = FakeAssetUploader()
        return ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
            asset_uploader=uploader,
        )

    def test_one_ref(self):
        backend = self._backend()
        wf = _native_r2v_workflow()
        patcher = WorkflowPatcher(wf)
        backend._patch_reference_images(patcher, ["/tmp/actor.png"])
        patched = patcher.get()
        self.assertIn("abc123", patched["50"]["inputs"]["image"])

    def test_two_refs(self):
        backend = self._backend()
        wf = _native_r2v_workflow()
        patcher = WorkflowPatcher(wf)
        backend._patch_reference_images(patcher, ["/tmp/actor.png", "/tmp/loc.png"])
        patched = patcher.get()
        self.assertIn("abc123", patched["50"]["inputs"]["image"])
        self.assertIn("loc", patched["60"]["inputs"]["image"])

    def test_no_paths_does_nothing(self):
        backend = self._backend()
        wf = _native_r2v_workflow()
        patcher = WorkflowPatcher(wf)
        backend._patch_reference_images(patcher, [])
        # Should not change any anchors
        self.assertEqual("placeholder.png", patcher.get()["50"]["inputs"]["image"])

    def test_none_does_nothing(self):
        backend = self._backend()
        wf = _native_r2v_workflow()
        patcher = WorkflowPatcher(wf)
        backend._patch_reference_images(patcher, None)
        self.assertEqual("placeholder.png", patcher.get()["50"]["inputs"]["image"])

    def test_max_refs_raises(self):
        backend = self._backend()
        wf = _native_r2v_workflow()
        patcher = WorkflowPatcher(wf)
        paths = [f"/tmp/ref{i}.png" for i in range(10)]
        with self.assertRaises(FeverSlopValidationError) as ctx:
            backend._patch_reference_images(patcher, paths)
        self.assertIn("9", str(ctx.exception))

    def test_anchors_beyond_workflow_count_ignored(self):
        """Only patch anchors that exist in the workflow."""
        backend = self._backend()
        # Workflow only has #REF_1 and #REF_2
        wf = _native_r2v_workflow()
        patcher = WorkflowPatcher(wf)
        backend._patch_reference_images(patcher, [
            "/tmp/a.png", "/tmp/b.png", "/tmp/c.png",
        ])
        # #REF_3 doesn't exist, so only 1 and 2 should be patched
        patched = patcher.get()
        self.assertIn("abc123", patched["50"]["inputs"]["image"])
        self.assertIn("b", patched["60"]["inputs"]["image"])


# ---------------------------------------------------------------------------
# _patch_audio_inputs tests
# ---------------------------------------------------------------------------

class PatchAudioInputsTests(unittest.TestCase):
    def test_audio_workflow_patches(self):
        wf = _audio_r2v_workflow()
        patcher = WorkflowPatcher(wf)
        ComfyUIMiniMaxH3R2VBackend._patch_audio_inputs(
            patcher, "my_audio-abc.wav", duration_seconds=8.0
        )
        patched = patcher.get()
        self.assertIn("my_audio", patched["90"]["inputs"]["audio"])
        self.assertIn("my_audio", patched["90"]["inputs"]["audioUI"])
        self.assertEqual(8.0, patched["91"]["inputs"]["duration"])

    def test_audio_workflow_no_duration(self):
        wf = _audio_r2v_workflow()
        patcher = WorkflowPatcher(wf)
        ComfyUIMiniMaxH3R2VBackend._patch_audio_inputs(
            patcher, "my_audio.wav", duration_seconds=None
        )
        patched = patcher.get()
        self.assertIn("my_audio", patched["90"]["inputs"]["audio"])
        # TRIM_AUDIO duration stays unchanged
        self.assertEqual(5.0, patched["91"]["inputs"]["duration"])

    def test_native_workflow_no_audio_anchors(self):
        """On a native workflow, the try_set_existing_input silently skips."""
        wf = _native_r2v_workflow()
        patcher = WorkflowPatcher(wf)
        # Should not raise, just silently not patch anything
        ComfyUIMiniMaxH3R2VBackend._patch_audio_inputs(
            patcher, "audio.wav", duration_seconds=5.0
        )
        # Workflow unchanged
        self.assertNotIn("90", patcher.get())


# ---------------------------------------------------------------------------
# _patch_seed tests
# ---------------------------------------------------------------------------

class PatchSeedTests(unittest.TestCase):
    def test_patches_noise_seed(self):
        wf = _native_r2v_workflow()
        patcher = WorkflowPatcher(wf)
        ComfyUIMiniMaxH3R2VBackend._patch_seed(patcher, 99999)
        self.assertEqual(99999, patcher.get()["20"]["inputs"]["noise_seed"])


# ---------------------------------------------------------------------------
# build_workflow tests
# ---------------------------------------------------------------------------

class BuildWorkflowTests(unittest.TestCase):
    def _backend(self, workflow: dict | None = None):
        uploader = FakeAssetUploader()
        return ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
            asset_uploader=uploader,
            workflow=workflow,
        )

    def test_full_flow(self):
        backend = self._backend(workflow=_native_r2v_workflow())
        result = backend.build_workflow(
            {"scene": 3, "references": {"actor_sheet_paths": ["/tmp/a.png"]}},
            prompt="cinematic shot",
            duration_seconds=5.0,
            width=1024,
            height=768,
            ref_image_paths=["/tmp/actor.png", "/tmp/loc.png"],
        )
        # Prompt patched
        self.assertEqual("cinematic shot", result["40"]["inputs"]["value"])
        # Duration patched
        self.assertEqual(5.0, result["30"]["inputs"]["value"])
        # Megapixels patched: round(1024 * 768 / 1_000_000, 1) = round(0.786432, 1) = 0.8
        expected_mp = round(1024 * 768 / 1_000_000, 1)
        self.assertEqual(expected_mp, result["10"]["inputs"]["megapixels"])
        # Ref images patched
        self.assertIn("actor", result["50"]["inputs"]["image"])
        self.assertIn("loc", result["60"]["inputs"]["image"])
        # Save prefix patched
        self.assertEqual("minimaxh3_raw/scene_0003", result["70"]["inputs"]["filename_prefix"])

    def test_seed_set(self):
        backend = self._backend(workflow=_native_r2v_workflow())
        backend.seed_offset = 50000
        backend.randomize_seed = False
        result = backend.build_workflow(
            {"scene": 7, "references": {"actor_sheet_paths": ["/tmp/a.png"]}},
            prompt="test",
        )
        self.assertEqual(50007, result["20"]["inputs"]["noise_seed"])

    def test_audio_included(self):
        backend = self._backend(workflow=_audio_r2v_workflow())
        result = backend.build_workflow(
            {"scene": 1, "references": {"actor_sheet_paths": ["/tmp/a.png"]}},
            prompt="test",
            comfy_audio_name="audio-xyz.wav",
            duration_seconds=10.0,
        )
        self.assertIn("audio-xyz", result["90"]["inputs"]["audio"])
        self.assertEqual(10.0, result["91"]["inputs"]["duration"])

    def test_no_audio_on_native_workflow(self):
        """Providing comfy_audio_name on a workflow without audio anchors is ok."""
        backend = self._backend(workflow=_native_r2v_workflow())
        result = backend.build_workflow(
            {"scene": 1, "references": {"actor_sheet_paths": ["/tmp/a.png"]}},
            prompt="test",
            comfy_audio_name="audio.wav",
        )
        # audio name was silently not patched
        self.assertNotIn("90", result)

    def test_validation_called(self):
        backend = self._backend(workflow=_native_r2v_workflow())
        with self.assertRaises(FeverSlopValidationError):
            backend.build_workflow(
                {"scene": 1},
                prompt="test",
            )


# ---------------------------------------------------------------------------
# render_video tests
# ---------------------------------------------------------------------------

class RenderVideoTests(unittest.TestCase):
    def test_full_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            uploader = FakeAssetUploader()
            queue = FakeRenderQueue()
            postprocessor = FakePostProcessor()
            resolver = FakeModelResolver()
            backend = ComfyUIMiniMaxH3R2VBackend(
                client=FakeClient(),
                workflow_path=Path("/tmp/wf.json"),
                output_dir=tmp_path / "output",
                asset_uploader=uploader,
                render_queue=queue,
                postprocessor=postprocessor,
                model_resolver=resolver,
                workflow=_native_r2v_workflow(),
            )
            request = VideoRenderRequest(
                scene={
                    "scene": 2,
                    "description": "A cat jumps",
                    "duration_seconds": 5.0,
                    "references": {"actor_sheet_paths": ["/tmp/actor.png"]},
                },
                scene_number=2,
                prompt="A cute cat jumps",
                workflow_path=Path("/tmp/wf.json"),
                output_dir=tmp_path / "output",
                audio_file=Path("/tmp/song.wav"),
                storyboard_dir=tmp_path / "storyboard",
                upload_audio=True,
            )
            result = backend.render_video(request)
            # Audio was resolved
            self.assertEqual(1, len(uploader.resolve_audio_calls))
            # Queue was called
            self.assertEqual(1, len(queue.calls))
            self.assertEqual(2, queue.calls[0]["scene_number"])
            # Postprocessor was called
            self.assertEqual(1, len(postprocessor.trim_specs))
            # Final file exists
            self.assertTrue(result.exists())
            self.assertEqual(tmp_path / "output" / "scene_0002.mp4", result)

    def test_no_postprocess_returns_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            uploader = FakeAssetUploader()
            queue = FakeRenderQueue()
            backend = ComfyUIMiniMaxH3R2VBackend(
                client=FakeClient(),
                workflow_path=Path("/tmp/wf.json"),
                output_dir=tmp_path / "output",
                asset_uploader=uploader,
                render_queue=queue,
                postprocess=False,
                model_resolver=FakeModelResolver(),
                workflow=_native_r2v_workflow(),
            )
            request = VideoRenderRequest(
                scene={
                    "scene": 1,
                    "description": "Test",
                    "references": {"actor_sheet_paths": ["/tmp/a.png"]},
                },
                scene_number=1,
                prompt="Test prompt",
                workflow_path=Path("/tmp/wf.json"),
                output_dir=tmp_path / "output",
                audio_file=Path("/tmp/song.wav"),
                storyboard_dir=tmp_path / "storyboard",
                upload_audio=False,
                uploaded_audio_name=None,
            )
            result = backend.render_video(request)
            self.assertIn("raw", str(result))

    def test_debug_workflow_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            uploader = FakeAssetUploader()
            queue = FakeRenderQueue()
            debug_dir = tmp_path / "debug"
            backend = ComfyUIMiniMaxH3R2VBackend(
                client=FakeClient(),
                workflow_path=Path("/tmp/wf.json"),
                output_dir=tmp_path / "output",
                asset_uploader=uploader,
                render_queue=queue,
                postprocess=False,
                model_resolver=FakeModelResolver(),
                debug_workflows_dir=debug_dir,
                workflow=_native_r2v_workflow(),
            )
            request = VideoRenderRequest(
                scene={
                    "scene": 1,
                    "description": "Test",
                    "references": {"actor_sheet_paths": ["/tmp/a.png"]},
                },
                scene_number=1,
                prompt="Test prompt",
                workflow_path=Path("/tmp/wf.json"),
                output_dir=tmp_path / "output",
                audio_file=Path("/tmp/song.wav"),
                storyboard_dir=tmp_path / "storyboard",
                upload_audio=False,
            )
            backend.render_video(request)
            debug_file = debug_dir / "scene_0001_workflow.json"
            self.assertTrue(debug_file.exists())
            data = json.loads(debug_file.read_text())
            self.assertIsInstance(data, dict)

    def test_validation_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = ComfyUIMiniMaxH3R2VBackend(
                client=FakeClient(),
                workflow_path=Path("/tmp/wf.json"),
                output_dir=Path(tmp),
                workflow=_native_r2v_workflow(),
            )
            request = VideoRenderRequest(
                scene={"scene": 1, "description": "Test"},
                scene_number=1,
                prompt="Test",
                workflow_path=Path("/tmp/wf.json"),
                output_dir=Path(tmp),
                audio_file=Path("/tmp/song.wav"),
                storyboard_dir=Path(tmp),
                upload_audio=False,
            )
            with self.assertRaises(FeverSlopValidationError):
                backend.render_video(request)


# ---------------------------------------------------------------------------
# _resolve_ref_image_paths tests
# ---------------------------------------------------------------------------

class ResolveRefImagePathsTests(unittest.TestCase):
    def test_actors_and_location(self):
        backend = ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
        )
        scene = {
            "references": {
                "actor_sheet_paths": ["actor1.png", "actor2.png"],
                "location_sheet_path": "loc.png",
            },
        }
        paths = backend._resolve_ref_image_paths(scene)
        self.assertEqual(3, len(paths))

    def test_clamped_to_max(self):
        backend = ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
        )
        scene = {
            "references": {
                "actor_sheet_paths": [f"a{i}.png" for i in range(10)],
            },
        }
        paths = backend._resolve_ref_image_paths(scene)
        self.assertEqual(9, len(paths))

    def test_empty_refs(self):
        backend = ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
        )
        scene = {"references": {}}
        paths = backend._resolve_ref_image_paths(scene)
        self.assertEqual(0, len(paths))


if __name__ == "__main__":
    unittest.main()
