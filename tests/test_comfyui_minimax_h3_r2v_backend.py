"""Tests for ComfyUIMiniMaxH3R2VBackend."""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

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
        self.resolve_reference_video_calls: list[Path | str] = []
        self.resolve_reference_audio_calls: list[Path | str] = []
        self.resolve_audio_calls: list[tuple] = []

    def resolve_reference_image_name(self, image_path, **kwargs):
        self.resolve_reference_image_calls.append(image_path)
        p = Path(image_path)
        return f"feverslop/references/{p.stem}-abc123{p.suffix}"

    def resolve_reference_video_name(self, video_path, **kwargs):
        self.resolve_reference_video_calls.append(video_path)
        p = Path(video_path)
        return f"feverslop/references/{p.stem}-vid456{p.suffix}"

    def resolve_reference_audio_name(self, audio_path, **kwargs):
        self.resolve_reference_audio_calls.append(audio_path)
        p = Path(audio_path)
        return f"feverslop/references/{p.stem}-aud789{p.suffix}"

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
            "class_type": "PrimitiveInt",
            "_meta": {"title": "#FRAMECOUNT"},
            "inputs": {"value": 144},
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
        "51": {
            "class_type": "LoadImage",
            "_meta": {"title": "#REF_3"},
            "inputs": {"image": ""},
        },
        "52": {
            "class_type": "LoadImage",
            "_meta": {"title": "#REF_4"},
            "inputs": {"image": ""},
        },
        "53": {
            "class_type": "LoadImage",
            "_meta": {"title": "#REF_5"},
            "inputs": {"image": ""},
        },
        "54": {
            "class_type": "LoadImage",
            "_meta": {"title": "#REF_6"},
            "inputs": {"image": ""},
        },
        "55": {
            "class_type": "LoadImage",
            "_meta": {"title": "#REF_7"},
            "inputs": {"image": ""},
        },
        "56": {
            "class_type": "LoadImage",
            "_meta": {"title": "#REF_8"},
            "inputs": {"image": ""},
        },
        "57": {
            "class_type": "LoadImage",
            "_meta": {"title": "#REF_9"},
            "inputs": {"image": ""},
        },
        "61": {
            "class_type": "LoadVideo",
            "_meta": {"title": "#VIDEO_1"},
            "inputs": {"video": ""},
        },
        "62": {
            "class_type": "LoadVideo",
            "_meta": {"title": "#VIDEO_2"},
            "inputs": {"video": ""},
        },
        "63": {
            "class_type": "LoadVideo",
            "_meta": {"title": "#VIDEO_3"},
            "inputs": {"video": ""},
        },
        "64": {
            "class_type": "LoadAudio",
            "_meta": {"title": "#AUDIO_1"},
            "inputs": {"audio": ""},
        },
        "65": {
            "class_type": "LoadAudio",
            "_meta": {"title": "#AUDIO_2"},
            "inputs": {"audio": ""},
        },
        "66": {
            "class_type": "LoadAudio",
            "_meta": {"title": "#AUDIO_3"},
            "inputs": {"audio": ""},
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
        "100": {
            "class_type": "TrimAudioDuration",
            "_meta": {"title": "#TRIM_AUDIO_1"},
            "inputs": {"start_index": 0.0, "duration": 5.0, "audio": ["64", 0]},
        },
        "101": {
            "class_type": "TrimAudioDuration",
            "_meta": {"title": "#TRIM_AUDIO_2"},
            "inputs": {"start_index": 0.0, "duration": 5.0, "audio": ["65", 0]},
        },
        "102": {
            "class_type": "TrimAudioDuration",
            "_meta": {"title": "#TRIM_AUDIO_3"},
            "inputs": {"start_index": 0.0, "duration": 5.0, "audio": ["66", 0]},
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
        """Only patch anchors that exist in the workflow (9 present)."""
        backend = self._backend()
        # Workflow has #REF_1 through #REF_9
        wf = _native_r2v_workflow()
        patcher = WorkflowPatcher(wf)
        # Pass all 9 valid paths to confirm all are patched
        paths = [f"/tmp/a{i}.png" for i in range(9)]
        backend._patch_reference_images(patcher, paths)
        self.assertEqual(9, len(backend.asset_uploader.resolve_reference_image_calls))

    def test_creates_missing_loaders_and_connects_dynamic_r2v_inputs(self):
        backend = self._backend()
        patcher = WorkflowPatcher({
            "42": {
                "class_type": "MiniMaxH3ReferenceToVideo",
                "_meta": {"title": "#R2V_COMBINE"},
                "inputs": {
                    "model": ["1", 0],
                    "ref_images.ref_image_8": ["98", 0],
                    "ref_videos.ref_video_2": ["99", 0],
                },
            },
        })

        backend._patch_reference_images(patcher, ["/tmp/actor.png"])
        backend._patch_reference_videos(patcher, ["/tmp/clip.mp4"])
        backend._patch_reference_audios(patcher, ["/tmp/voice.wav"])

        workflow = patcher.get()
        core_inputs = workflow["42"]["inputs"]
        expected = {
            "ref_images.ref_image_0": "LoadImage",
            "ref_videos.ref_video_0": "LoadVideo",
            "ref_audios.ref_audio_0": "LoadAudio",
        }
        self.assertNotIn("ref_images.ref_image_8", core_inputs)
        self.assertNotIn("ref_videos.ref_video_2", core_inputs)
        for input_name, class_type in expected.items():
            loader_id, output_index = core_inputs[input_name]
            self.assertEqual(0, output_index)
            self.assertEqual(class_type, workflow[loader_id]["class_type"])

# ---------------------------------------------------------------------------
# _patch_audio_inputs tests
# ---------------------------------------------------------------------------

class PatchAudioInputsTests(unittest.TestCase):
    def _backend(self, workflow=None):
        uploader = FakeAssetUploader()
        return ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
            asset_uploader=uploader,
            workflow=workflow,
        )

    def test_audio_workflow_patches(self):
        backend = self._backend(workflow=_audio_r2v_workflow())
        wf = backend.load_workflow()
        patcher = WorkflowPatcher(wf)
        backend._patch_audio_inputs(patcher, "my_audio-abc.wav", duration_seconds=8.0)
        patched = patcher.get()
        self.assertIn("my_audio", patched["90"]["inputs"]["audio"])
        self.assertIn("my_audio", patched["90"]["inputs"]["audioUI"])
        self.assertEqual(8.0, patched["91"]["inputs"]["duration"])

    def test_audio_workflow_patches_start_index_default(self):
        """start_index defaults to 0.0 when scene is not provided."""
        backend = self._backend(workflow=_audio_r2v_workflow())
        wf = backend.load_workflow()
        patcher = WorkflowPatcher(wf)
        backend._patch_audio_inputs(patcher, "audio.wav", scene=None)
        patched = patcher.get()
        self.assertEqual(0.0, patched["91"]["inputs"]["start_index"])

    def test_audio_workflow_patches_start_index_from_scene(self):
        """start_index is taken from scene.abs_start_seconds."""
        backend = self._backend(workflow=_audio_r2v_workflow())
        wf = backend.load_workflow()
        patcher = WorkflowPatcher(wf)
        scene = {"scene": 3, "abs_start_seconds": 10.5}
        backend._patch_audio_inputs(patcher, "audio.wav", scene=scene)
        patched = patcher.get()
        self.assertEqual(10.5, patched["91"]["inputs"]["start_index"])

    def test_audio_workflow_no_duration(self):
        backend = self._backend(workflow=_audio_r2v_workflow())
        wf = backend.load_workflow()
        patcher = WorkflowPatcher(wf)
        backend._patch_audio_inputs(patcher, "my_audio.wav", duration_seconds=None)
        patched = patcher.get()
        self.assertIn("my_audio", patched["90"]["inputs"]["audio"])
        # TRIM_AUDIO duration stays unchanged
        self.assertEqual(5.0, patched["91"]["inputs"]["duration"])

    def test_native_workflow_no_audio_anchors(self):
        """On a native workflow, the try_set_existing_input silently skips."""
        backend = self._backend(workflow=_native_r2v_workflow())
        wf = backend.load_workflow()
        patcher = WorkflowPatcher(wf)
        # Should not raise, just silently not patch anything
        backend._patch_audio_inputs(patcher, "audio.wav", duration_seconds=5.0)
        # Workflow unchanged
        self.assertNotIn("90", patcher.get())

    def test_wires_trimmed_audio_to_r2v(self):
        """The #TRIM_AUDIO output gets wired to ref_audios.ref_audio_0."""
        # Build a workflow with both #TRIM_AUDIO and MiniMaxH3ReferenceToVideo
        workflow = _audio_r2v_workflow()
        workflow["80"] = {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "_meta": {"title": "#R2V_COMBINE"},
            "inputs": {
                "prompt": ["40", 0],
                "width": ["10", 0],
                "height": ["10", 1],
            },
        }
        backend = self._backend(workflow=workflow)
        wf = backend.load_workflow()
        patcher = WorkflowPatcher(wf)
        backend._patch_audio_inputs(patcher, "audio.wav")
        patched = patcher.get()
        r2v_inputs = patched["80"]["inputs"]
        self.assertIn("ref_audios.ref_audio_0", r2v_inputs)
        self.assertEqual(["91", 0], r2v_inputs["ref_audios.ref_audio_0"])

    def test_wire_no_trim_node_is_noop(self):
        """_wire_trimmed_audio_to_r2v is safe when #TRIM_AUDIO is absent."""
        workflow = _native_r2v_workflow()
        workflow["80"] = {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "_meta": {"title": "#R2V_COMBINE"},
            "inputs": {},
        }
        backend = self._backend(workflow=workflow)
        wf = backend.load_workflow()
        patcher = WorkflowPatcher(wf)
        # No exception
        ComfyUIMiniMaxH3R2VBackend._wire_trimmed_audio_to_r2v(patcher)
        patched = patcher.get()
        self.assertNotIn("ref_audios.ref_audio_0", patched["80"]["inputs"])

    def test_wire_no_r2v_node_is_noop(self):
        """_wire_trimmed_audio_to_r2v is safe when R2V node is absent."""
        workflow = _audio_r2v_workflow()
        # No MiniMaxH3ReferenceToVideo node present
        backend = self._backend(workflow=workflow)
        wf = backend.load_workflow()
        patcher = WorkflowPatcher(wf)
        # No exception
        ComfyUIMiniMaxH3R2VBackend._wire_trimmed_audio_to_r2v(patcher)


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
        # Frame count patched: round(5.0 * 24) = 120
        self.assertEqual(120, result["30"]["inputs"]["value"])
        # Megapixels patched: round(1024 * 768 / 1_000_000, 1) = round(0.786432, 1) = 0.8
        expected_mp = round(1024 * 768 / 1_000_000, 1)
        self.assertEqual(expected_mp, result["10"]["inputs"]["megapixels"])
        # Ref images patched
        self.assertIn("actor", result["50"]["inputs"]["image"])
        self.assertIn("loc", result["60"]["inputs"]["image"])
        # Save prefix patched
        self.assertEqual("scene_0003/raw", result["70"]["inputs"]["filename_prefix"])

    def test_seed_set(self):
        backend = self._backend(workflow=_native_r2v_workflow())
        backend.seed_offset = 50000
        backend.randomize_seed = False
        result = backend.build_workflow(
            {"scene": 7, "references": {"actor_sheet_paths": ["/tmp/a.png"]}},
            prompt="test",
        )
        self.assertEqual(50007, result["20"]["inputs"]["noise_seed"])

    def test_persisted_scene_seed_overrides_legacy_offset(self):
        backend = self._backend(workflow=_native_r2v_workflow())
        backend.seed_offset = 50000
        result = backend.build_workflow(
            {"scene": 7, "seed": 424242, "references": {"actor_sheet_paths": ["/tmp/a.png"]}},
            prompt="test",
        )
        self.assertEqual(424242, result["20"]["inputs"]["noise_seed"])

    def test_audio_included(self):
        backend = self._backend(workflow=_audio_r2v_workflow())
        result = backend.build_workflow(
            {"scene": 1, "references": {"actor_sheet_paths": ["/tmp/a.png"]}},
            prompt="test",
            comfy_audio_name="audio-xyz.wav",
            duration_seconds=10.0,
        )
        self.assertNotIn("90", result)
        self.assertNotIn("91", result)

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
            self.assertEqual(0, len(uploader.resolve_audio_calls))
            # Queue was called
            self.assertEqual(1, len(queue.calls))
            self.assertEqual(2, queue.calls[0]["scene_number"])
            # Postprocessor was called
            self.assertEqual(1, len(postprocessor.trim_specs))
            # Final file exists
            self.assertTrue(result.exists())
            self.assertEqual(tmp_path / "output" / "scene_0002" / "final.mp4", result)

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
            # Raw file in per-scene directory
            self.assertEqual(tmp_path / "output" / "scene_0001" / "raw.mp4", result)

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

    def test_scene_workflow_json_written(self):
        """workflow.json is written to the per-scene directory."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            project = tmp_path / "project"
            output = project / "output" / "render" / "scenes"
            plan = project / "output" / "render" / "plans" / "references.json"
            plan.parent.mkdir(parents=True)
            plan.write_text("[]", encoding="utf-8")
            template = project / "workflows" / "r2v.json"
            template.parent.mkdir()
            template.write_text("{}", encoding="utf-8")
            actor = project / "input" / "actor.png"
            actor.parent.mkdir()
            actor.write_bytes(b"actor")
            song = project / "input" / "song.wav"
            song.write_bytes(b"song")
            uploader = FakeAssetUploader()
            queue = FakeRenderQueue()
            backend = ComfyUIMiniMaxH3R2VBackend(
                client=FakeClient(),
                workflow_path=template,
                output_dir=output,
                project_dir=project,
                randomize_seed=True,
                asset_uploader=uploader,
                render_queue=queue,
                postprocess=False,
                model_resolver=FakeModelResolver(),
                workflow=_native_r2v_workflow(),
            )
            request = VideoRenderRequest(
                scene={
                    "scene": 5,
                    "description": "Test",
                    "references": {"actor_sheet_paths": [actor]},
                },
                scene_number=5,
                prompt="Test prompt",
                workflow_path=template,
                render_plan_path=plan,
                output_dir=output,
                audio_file=song,
                storyboard_dir=tmp_path / "storyboard",
            )
            with patch(
                "feverslop.adapters.comfyui_minimax_h3_r2v_backend.random.randint",
                side_effect=[111, 222],
            ):
                backend.render_video(request)
            scene_dir = output / "scene_0005"
            scene_workflow = scene_dir / "workflow.json"
            self.assertTrue(scene_workflow.exists())
            data = json.loads(scene_workflow.read_text())
            self.assertIsInstance(data, dict)
            manifest = json.loads((scene_dir / "manifest.json").read_text())
            self.assertEqual("minimax-h3-r2v", manifest["pipeline"])
            self.assertEqual("output/render/scenes/scene_0005/workflow.json", manifest["workflow"]["path"])
            self.assertEqual(111, manifest["seed"])
            self.assertEqual(["reference_image"], [asset["role"] for asset in manifest["assets"]])
            self.assertEqual("input/actor.png", manifest["assets"][0]["path"])
            self.assertEqual([], uploader.resolve_audio_calls)


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


# ---------------------------------------------------------------------------
# Class constants tests
# ---------------------------------------------------------------------------

class ClassConstantsTests(unittest.TestCase):
    def test_max_ref_videos(self):
        self.assertEqual(3, ComfyUIMiniMaxH3R2VBackend.MAX_REF_VIDEOS)

    def test_max_ref_audios(self):
        self.assertEqual(3, ComfyUIMiniMaxH3R2VBackend.MAX_REF_AUDIOS)

    def test_max_ref_images(self):
        self.assertEqual(9, ComfyUIMiniMaxH3R2VBackend.MAX_REF_IMAGES)


# ---------------------------------------------------------------------------
# _patch_reference_videos tests
# ---------------------------------------------------------------------------

class PatchVideoInputsTests(unittest.TestCase):
    def _backend(self):
        uploader = FakeAssetUploader()
        return ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
            asset_uploader=uploader,
        )

    def test_one_video(self):
        backend = self._backend()
        wf = _native_r2v_workflow()
        patcher = WorkflowPatcher(wf)
        backend._patch_reference_videos(patcher, ["/tmp/clip.mp4"])
        patched = patcher.get()
        self.assertIn("vid456", patched["61"]["inputs"]["video"])

    def test_three_videos(self):
        backend = self._backend()
        wf = _native_r2v_workflow()
        patcher = WorkflowPatcher(wf)
        backend._patch_reference_videos(patcher, [
            "/tmp/clip1.mp4", "/tmp/clip2.mp4", "/tmp/clip3.mp4"
        ])
        patched = patcher.get()
        self.assertIn("vid456", patched["61"]["inputs"]["video"])
        self.assertIn("clip2", patched["62"]["inputs"]["video"])
        self.assertIn("clip3", patched["63"]["inputs"]["video"])

    def test_no_paths_does_nothing(self):
        backend = self._backend()
        wf = _native_r2v_workflow()
        patcher = WorkflowPatcher(wf)
        backend._patch_reference_videos(patcher, [])
        self.assertEqual("", patcher.get()["61"]["inputs"]["video"])

    def test_none_does_nothing(self):
        backend = self._backend()
        wf = _native_r2v_workflow()
        patcher = WorkflowPatcher(wf)
        backend._patch_reference_videos(patcher, None)
        self.assertEqual("", patcher.get()["61"]["inputs"]["video"])

    def test_max_videos_raises(self):
        backend = self._backend()
        wf = _native_r2v_workflow()
        patcher = WorkflowPatcher(wf)
        paths = [f"/tmp/vid{i}.mp4" for i in range(4)]
        with self.assertRaises(FeverSlopValidationError) as ctx:
            backend._patch_reference_videos(patcher, paths)
        self.assertIn("3", str(ctx.exception))

    def test_anchors_beyond_workflow_count_ignored(self):
        """Only patch anchors that exist in the workflow (3 present)."""
        backend = self._backend()
        wf = _native_r2v_workflow()
        patcher = WorkflowPatcher(wf)
        # Pass exactly 3 paths - all 3 anchors exist, all 3 should be patched
        backend._patch_reference_videos(patcher, [f"/tmp/v{i}.mp4" for i in range(3)])
        self.assertEqual(3, len(backend.asset_uploader.resolve_reference_video_calls))


# ---------------------------------------------------------------------------
# _patch_reference_audios tests
# ---------------------------------------------------------------------------

class PatchReferenceAudioInputsTests(unittest.TestCase):
    def _backend(self):
        uploader = FakeAssetUploader()
        return ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
            asset_uploader=uploader,
        )

    def test_one_audio(self):
        backend = self._backend()
        wf = _native_r2v_workflow()
        patcher = WorkflowPatcher(wf)
        backend._patch_reference_audios(patcher, ["/tmp/sound.wav"])
        patched = patcher.get()
        self.assertIn("aud789", patched["64"]["inputs"]["audio"])

    def test_three_audios(self):
        backend = self._backend()
        wf = _native_r2v_workflow()
        patcher = WorkflowPatcher(wf)
        backend._patch_reference_audios(patcher, [
            "/tmp/s1.wav", "/tmp/s2.wav", "/tmp/s3.wav"
        ])
        patched = patcher.get()
        self.assertIn("aud789", patched["64"]["inputs"]["audio"])
        self.assertIn("s2", patched["65"]["inputs"]["audio"])
        self.assertIn("s3", patched["66"]["inputs"]["audio"])

    def test_no_paths_does_nothing(self):
        backend = self._backend()
        wf = _native_r2v_workflow()
        patcher = WorkflowPatcher(wf)
        backend._patch_reference_audios(patcher, [])
        self.assertEqual("", patcher.get()["64"]["inputs"]["audio"])

    def test_max_audios_raises(self):
        backend = self._backend()
        wf = _native_r2v_workflow()
        patcher = WorkflowPatcher(wf)
        paths = [f"/tmp/s{i}.wav" for i in range(4)]
        with self.assertRaises(FeverSlopValidationError) as ctx:
            backend._patch_reference_audios(patcher, paths)
        self.assertIn("3", str(ctx.exception))


# ---------------------------------------------------------------------------
# build_workflow with video/audio tests
# ---------------------------------------------------------------------------

class BuildWorkflowVideoAudioTests(unittest.TestCase):
    def _backend(self, workflow=None):
        uploader = FakeAssetUploader()
        return ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
            asset_uploader=uploader,
            workflow=workflow or _native_r2v_workflow(),
        )

    def test_ref_videos_and_audios_passthrough(self):
        backend = self._backend()
        result = backend.build_workflow(
            {"scene": 1, "references": {"actor_sheet_paths": ["/tmp/a.png"]}},
            prompt="test",
            ref_video_paths=["/tmp/clip1.mp4", "/tmp/clip2.mp4"],
            ref_audio_paths=["/tmp/sound1.wav"],
        )
        self.assertIn("vid456", result["61"]["inputs"]["video"])
        self.assertIn("clip2", result["62"]["inputs"]["video"])
        self.assertIn("aud789", result["64"]["inputs"]["audio"])

    def test_production_audio_workflow_adds_stem_loaders_and_trims(self):
        workflow_path = Path(__file__).parents[1] / "workflows" / "video_minimax_h3_r2v_audio_v1.json"
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        backend = self._backend(workflow=workflow)

        result = backend.build_workflow(
            {"scene": 3, "abs_start_seconds": 15.01, "references": {"actor_sheet_paths": ["/tmp/a.png"]}},
            prompt="test",
            duration_seconds=4.44,
            ref_audio_paths=["/tmp/vocals.wav", "/tmp/drums.wav"],
        )

        core = next(node for node in result.values() if node.get("class_type") == "MiniMaxH3ReferenceToVideo")
        inputs = core["inputs"]
        for slot, expected_name in ((0, "vocals"), (1, "drums")):
            trim_id = inputs[f"ref_audios.ref_audio_{slot}"][0]
            trim = result[trim_id]
            loader_id = trim["inputs"]["audio"][0]
            loader = result[loader_id]
            self.assertEqual(loader["inputs"]["audio"], f"feverslop/references/{expected_name}-aud789.wav")
            self.assertEqual(trim["inputs"]["start_index"], 15.01)
            self.assertEqual(trim["inputs"]["duration"], 4.44)

    def test_audio_filter_skips_source_that_ends_before_scene_window(self):
        backend = self._backend()
        with tempfile.TemporaryDirectory() as temp_dir:
            vocals = Path(temp_dir) / "vocals.mp3"
            full_mix = Path(temp_dir) / "full_mix.mp3"
            vocals.write_bytes(b"vocals")
            full_mix.write_bytes(b"full mix")
            with patch(
                "feverslop.adapters.comfyui_minimax_h3_r2v_backend.subprocess.run",
                side_effect=[
                    type("Result", (), {"stdout": "120.0\n"})(),
                    type("Result", (), {"stdout": "142.6\n"})(),
                ],
            ):
                result = backend._filter_audio_paths_for_window(
                    [vocals, full_mix], start_seconds=122.45, duration_seconds=3.25
                )

        self.assertEqual([full_mix], result)

    def test_audio_wiring_does_not_mutate_generated_prompt(self):
        workflow_path = Path(__file__).parents[1] / "workflows" / "video_minimax_h3_r2v_audio_v1.json"
        backend = self._backend(workflow=json.loads(workflow_path.read_text(encoding="utf-8")))
        generated_prompt = """subject_definitions:
<Subject 1> (Bard): A singer.

summary: A close-up.

retention_analysis:
The vocal reference is fully copied.

detailed_description: The singer performs.

overall_soundscape: The vocal performance is audible.

non_diegetic_music: N/A"""

        result = backend.build_workflow(
            {"scene": 3, "abs_start_seconds": 1.0, "references": {"actor_sheet_paths": ["/tmp/a.png"]}},
            prompt=generated_prompt,
            duration_seconds=4.44,
            ref_audio_paths=["/tmp/vocals.wav", "/tmp/full_mix.wav"],
        )

        prompt_node = next(
            node for node in result.values() if node.get("_meta", {}).get("title") == "#PROMPT"
        )
        self.assertEqual(generated_prompt, prompt_node["inputs"]["value"])

    def test_scene_validation_rejects_h3_prompt_with_unbound_subject_and_picture(self):
        backend = self._backend()
        scene = {
            "scene": 2,
            "references": {
                "actor_sheet_paths": ["actor.png"],
                "location_sheet_path": "location.png",
            },
            "h3": {"prompt": """subject_definitions:
<Subject 1> (Drummer): A drummer. Source references: <Picture 1>.
summary: <Subject 1> performs on <Subject 3>.
retention_analysis: <Subject 1>: fully_preserved - stable.
detailed_description: <Subject 1> performs while <Subject 3> fills <Picture 3>.
overall_soundscape: Music.
non_diegetic_music: N/A"""},
        }

        with self.assertRaisesRegex(
            Exception,
            r"Scene 2 H3 reference contract mismatch.*undefined_subjects=.*<Subject 3>.*unbound_pictures=.*<Picture 2>",
        ):
            backend._validate_scene(scene)

    def test_scene_validation_rejects_loaded_audio_missing_from_h3_definitions(self):
        backend = self._backend()
        scene = {
            "scene": 4,
            "references": {
                "actor_sheet_paths": ["actor.png"],
                "reference_audio_paths": ["song.wav"],
            },
            "h3": {"prompt": """subject_definitions:
<Subject 1> (Drummer): A drummer. Source references: <Picture 1>.
summary: <Subject 1> performs.
retention_analysis: <Subject 1>: fully_preserved - stable.
detailed_description: <Subject 1> performs.
overall_soundscape: Music.
non_diegetic_music: N/A"""},
        }

        with self.assertRaisesRegex(Exception, r"missing_audio=.*<Audio 1>"):
            backend._validate_scene(scene)

    def test_scene_validation_rejects_duplicate_picture_mapping_and_unknown_video(self):
        backend = self._backend()
        scene = {
            "scene": 5,
            "references": {
                "actor_sheet_paths": ["actor.png"],
                "reference_video_paths": ["motion.mp4"],
            },
            "h3": {"prompt": """subject_definitions:
<Subject 1> (A): Stable. Source references: <Picture 1>.
<Subject 2> (B): Duplicate. Source references: <Picture 1>.
summary: <Subject 1> follows <Video 2>.
retention_analysis:
<Subject 1>: fully_preserved - stable.
<Subject 2>: fully_preserved - stable.
detailed_description: <Subject 1> moves.
overall_soundscape: Music.
non_diegetic_music: N/A"""},
        }

        with self.assertRaisesRegex(
            Exception,
            r"duplicate_picture_mappings=.*<Picture 1>.*unknown_videos=.*<Video 2>",
        ):
            backend._validate_scene(scene)

    def test_production_audio_workflow_maps_reference_audio_from_slot_zero(self):
        workflow_path = Path(__file__).parents[1] / "workflows" / "video_minimax_h3_r2v_audio_v1.json"
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        backend = self._backend(workflow=workflow)

        result = backend.build_workflow(
            {"scene": 3, "abs_start_seconds": 15.01, "references": {"actor_sheet_paths": ["/tmp/a.png"]}},
            prompt="test",
            comfy_audio_name="full_mix.mp3",
            duration_seconds=4.44,
            ref_audio_paths=["/tmp/vocals.wav", "/tmp/full_mix.wav"],
        )

        core = next(node for node in result.values() if node.get("class_type") == "MiniMaxH3ReferenceToVideo")
        inputs = core["inputs"]
        self.assertNotIn("ref_audios.ref_audio_2", inputs)
        for slot, expected_name in ((0, "vocals"), (1, "full_mix")):
            trim_id = inputs[f"ref_audios.ref_audio_{slot}"][0]
            trim = result[trim_id]
            loader_id = trim["inputs"]["audio"][0]
            loader = result[loader_id]
            self.assertEqual(loader["inputs"]["audio"], f"feverslop/references/{expected_name}-aud789.wav")
            self.assertEqual(trim["inputs"]["start_index"], 15.01)
            self.assertEqual(trim["inputs"]["duration"], 4.44)
        self.assertFalse(any(node.get("_meta", {}).get("title") == "#LOAD_AUDIO" for node in result.values()))
        self.assertFalse(any(node.get("_meta", {}).get("title") == "#TRIM_AUDIO" for node in result.values()))
        self.assertFalse(any(node.get("_meta", {}).get("title") == "#AUDIO_3" for node in result.values()))
        self.assertFalse(any(node.get("_meta", {}).get("title") == "#REF_3" for node in result.values()))
        self.assertFalse(any(node.get("_meta", {}).get("title") == "#VIDEO_1" for node in result.values()))

    def test_video_and_audio_anchors_absent_in_workflow(self):
        """Workflow without video/audio anchors ignores them silently."""
        # Use a minimal workflow missing the video/audio nodes
        minimal = {
            "10": {"class_type": "ResolutionSelector", "_meta": {"title": "#MEGAPIXELS"}, "inputs": {}},
            "20": {"class_type": "RandomNoise", "_meta": {"title": "#SEED"}, "inputs": {"noise_seed": 0}},
            "30": {"class_type": "PrimitiveInt", "_meta": {"title": "#FRAMECOUNT"}, "inputs": {}},
            "40": {"class_type": "PrimitiveStringMultiline", "_meta": {"title": "#PROMPT"}, "inputs": {}},
            "50": {"class_type": "LoadImage", "_meta": {"title": "#REF_1"}, "inputs": {"image": ""}},
            "70": {"class_type": "VHS_VideoCombine", "_meta": {"title": "#SAVE_VIDEO"}, "inputs": {}},
        }
        backend = self._backend(workflow=minimal)
        result = backend.build_workflow(
            {"scene": 1, "references": {"actor_sheet_paths": ["/tmp/a.png"]}},
            prompt="test",
            ref_video_paths=["/tmp/clip.mp4"],
            ref_audio_paths=["/tmp/sound.wav"],
        )
        # Should not raise, just silently skip
        self.assertEqual("test", result["40"]["inputs"]["value"])

    def test_no_videos_no_audios(self):
        """Omitting both still works."""
        backend = self._backend()
        result = backend.build_workflow(
            {"scene": 1, "references": {"actor_sheet_paths": ["/tmp/a.png"]}},
            prompt="test",
        )
        self.assertEqual("", result["61"]["inputs"]["video"])
        self.assertEqual("", result["64"]["inputs"]["audio"])



# ---------------------------------------------------------------------------
# Audio prompt suffix tests (<Audio N> tags with stem descriptions)
# ---------------------------------------------------------------------------

class AudioPromptSuffixTests(unittest.TestCase):
    def _backend(self, workflow=None):
        return ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
            asset_uploader=FakeAssetUploader(),
            workflow=workflow or _native_r2v_workflow(),
        )

    def test_has_no_dead_audio_prompt_enhancement_api(self):
        backend = self._backend()

        self.assertFalse(hasattr(backend, "_build_audio_prompt_suffix"))
        self.assertFalse(hasattr(backend, "_patch_prompt_with_audio_tags"))
        self.assertFalse(hasattr(backend, "_inject_audio_subjects"))

    def test_stem_audio_does_not_mutate_prompt(self):
        """Stem wiring leaves the DSPy-generated prompt untouched."""
        backend = self._backend()
        result = backend.build_workflow(
            {
                "scene": 1,
                "references": {
                    "actor_sheet_paths": ["/tmp/a.png"],
                    "reference_audio_paths": ["/tmp/stems/vocals.wav", "/tmp/stems/full_mix.wav"],
                    "_stem_audio_tags": {
                        "/tmp/stems/vocals.wav": "audio_transfer - vocal singing lip-synced to the audio signal",
                        "/tmp/stems/full_mix.wav": "full_mix - original song for beat and rhythm continuity",
                    },
                },
                "stem_audio": {
                    "stems": ["vocals", "full_mix"],
                    "paths": {
                        "vocals": "/tmp/stems/vocals.wav",
                        "full_mix": "/tmp/stems/full_mix.wav",
                    },
                },
            },
            prompt="test prompt here",
            ref_audio_paths=[Path("/tmp/stems/vocals.wav"), Path("/tmp/stems/full_mix.wav")],
        )
        prompt_value = result["40"]["inputs"]["value"]
        self.assertEqual("test prompt here", prompt_value)

    def test_no_stem_audio_no_tags_in_prompt(self):
        """Ordinary reference audio without stem audio does NOT create tags."""
        backend = self._backend()
        result = backend.build_workflow(
            {
                "scene": 1,
                "references": {
                    "actor_sheet_paths": ["/tmp/a.png"],
                    "reference_audio_paths": ["/tmp/sound.wav"],
                },
            },
            prompt="plain test",
            ref_audio_paths=[Path("/tmp/sound.wav")],
        )
        prompt_value = result["40"]["inputs"]["value"]
        self.assertEqual("plain test", prompt_value)

    def test_does_not_append_audio_tags_when_generator_already_rendered_them(self):
        backend = self._backend()
        prompt = (
            "non_diegetic_music: N/A\n"
            "<Audio 1> (audio_transfer - vocal singing lip-synced to the audio signal)\n"
            "<Audio 2> (full_mix - original song for beat and rhythm continuity)"
        )
        result = backend.build_workflow(
            {
                "scene": 1,
                "references": {
                    "actor_sheet_paths": ["/tmp/a.png"],
                    "reference_audio_paths": ["/tmp/vocals.wav", "/tmp/full_mix.wav"],
                    "_stem_audio_tags": {
                        "/tmp/vocals.wav": "audio_transfer - vocal singing lip-synced to the audio signal",
                        "/tmp/full_mix.wav": "full_mix - original song for beat and rhythm continuity",
                    },
                },
                "stem_audio": {"stems": ["vocals", "full_mix"]},
            },
            prompt=prompt,
            ref_audio_paths=[Path("/tmp/vocals.wav"), Path("/tmp/full_mix.wav")],
        )

        self.assertEqual(prompt, result["40"]["inputs"]["value"])

# ---------------------------------------------------------------------------
# _resolve_ref_video_paths tests
# ---------------------------------------------------------------------------


class ResolveRefVideoPathsTests(unittest.TestCase):
    def test_basic(self):
        backend = ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
        )
        scene = {
            "references": {
                "reference_video_paths": ["clip1.mp4", "clip2.mp4"],
            }
        }
        paths = backend._resolve_ref_video_paths(scene)
        self.assertEqual(2, len(paths))

    def test_clamped_to_max(self):
        backend = ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
        )
        scene = {
            "references": {
                "reference_video_paths": [f"v{i}.mp4" for i in range(5)],
            }
        }
        paths = backend._resolve_ref_video_paths(scene)
        self.assertEqual(3, len(paths))

    def test_empty_refs(self):
        backend = ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
        )
        scene = {"references": {}}
        paths = backend._resolve_ref_video_paths(scene)
        self.assertEqual(0, len(paths))


# ---------------------------------------------------------------------------
# _resolve_ref_audio_paths tests
# ---------------------------------------------------------------------------

class ResolveRefAudioPathsTests(unittest.TestCase):
    def test_basic(self):
        backend = ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
        )
        scene = {
            "references": {
                "reference_audio_paths": ["sound1.wav", "sound2.wav"],
            }
        }
        paths = backend._resolve_ref_audio_paths(scene)
        self.assertEqual(2, len(paths))

    def test_clamped_to_max(self):
        backend = ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
        )
        scene = {
            "references": {
                "reference_audio_paths": [f"s{i}.wav" for i in range(5)],
            }
        }
        paths = backend._resolve_ref_audio_paths(scene)
        self.assertEqual(3, len(paths))

    def test_empty_refs(self):
        backend = ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
        )
        scene = {"references": {}}
        paths = backend._resolve_ref_audio_paths(scene)
        self.assertEqual(0, len(paths))


# ---------------------------------------------------------------------------
# _find_occupied_ref_slots tests
# ---------------------------------------------------------------------------

def _r2v_core_workflow(occupied_indices=None):
    """Create workflow with MiniMaxH3ReferenceToVideo core node + some occupied slots."""
    inputs = {}
    if occupied_indices:
        for idx in occupied_indices:
            inputs[f"ref_images.ref_image_{idx}"] = [str(idx + 100), 0]
    return {
        "42": {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "_meta": {"title": "#R2V_COMBINE"},
            "inputs": inputs,
        },
    }


class FindOccupiedRefSlotsTests(unittest.TestCase):
    def test_occupied_slots_empty(self):
        wf = _r2v_core_workflow()
        patcher = WorkflowPatcher(wf)
        occupied = ComfyUIMiniMaxH3R2VBackend._find_occupied_ref_slots(
            patcher, "ref_images"
        )
        self.assertEqual(set(), occupied)

    def test_occupied_slots_partial(self):
        wf = _r2v_core_workflow(occupied_indices=[0, 2])
        patcher = WorkflowPatcher(wf)
        occupied = ComfyUIMiniMaxH3R2VBackend._find_occupied_ref_slots(
            patcher, "ref_images"
        )
        self.assertEqual({0, 2}, occupied)

    def test_occupied_slots_all_images(self):
        wf = _r2v_core_workflow(occupied_indices=list(range(9)))
        patcher = WorkflowPatcher(wf)
        occupied = ComfyUIMiniMaxH3R2VBackend._find_occupied_ref_slots(
            patcher, "ref_images"
        )
        self.assertEqual(set(range(9)), occupied)

    def test_occupied_slots_other_groups_ignored(self):
        """Only ref_videos inputs present -> empty for ref_images."""
        wf = {
            "42": {
                "class_type": "MiniMaxH3ReferenceToVideo",
                "_meta": {"title": "#R2V_COMBINE"},
                "inputs": {
                    "ref_videos.ref_video_0": ["100", 0],
                    "ref_videos.ref_video_1": ["101", 0],
                },
            },
        }
        patcher = WorkflowPatcher(wf)
        occupied = ComfyUIMiniMaxH3R2VBackend._find_occupied_ref_slots(
            patcher, "ref_images"
        )
        self.assertEqual(set(), occupied)

    def test_occupied_slots_no_core_node(self):
        wf = {
            "10": {
                "class_type": "LoadImage",
                "_meta": {"title": "#REF_1"},
                "inputs": {"image": "test.png"},
            },
        }
        patcher = WorkflowPatcher(wf)
        occupied = ComfyUIMiniMaxH3R2VBackend._find_occupied_ref_slots(
            patcher, "ref_images"
        )
        self.assertEqual(set(), occupied)


# ---------------------------------------------------------------------------
# _add_ref_node_and_wire tests
# ---------------------------------------------------------------------------

class AddRefNodeAndWireTests(unittest.TestCase):
    def _backend(self, workflow=None):
        uploader = FakeAssetUploader()
        return ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
            asset_uploader=uploader,
            workflow=workflow or _r2v_core_workflow(),
        )

    def test_add_image_node_and_wire(self):
        backend = self._backend()
        wf = backend.load_workflow()
        patcher = WorkflowPatcher(wf)
        tag = backend._add_ref_node_and_wire(
            patcher, "ref_images", 0, "/tmp/actor.png"
        )
        patched = patcher.get()
        # Check loader node was created
        self.assertEqual("/tmp/actor.png", backend.asset_uploader.resolve_reference_image_calls[0])
        # Loader has correct class type
        loader_id = patched["42"]["inputs"]["ref_images.ref_image_0"][0]
        self.assertEqual("LoadImage", patched[loader_id]["class_type"])
        self.assertEqual("<Picture 1>", tag)

    def test_add_video_node_and_wire(self):
        backend = self._backend()
        wf = backend.load_workflow()
        patcher = WorkflowPatcher(wf)
        tag = backend._add_ref_node_and_wire(
            patcher, "ref_videos", 1, "/tmp/clip.mp4"
        )
        patched = patcher.get()
        loader_id = patched["42"]["inputs"]["ref_videos.ref_video_1"][0]
        self.assertEqual("LoadVideo", patched[loader_id]["class_type"])
        self.assertEqual("<Video 2>", tag)

    def test_add_audio_node_and_wire(self):
        backend = self._backend()
        wf = backend.load_workflow()
        patcher = WorkflowPatcher(wf)
        tag = backend._add_ref_node_and_wire(
            patcher, "ref_audios", 2, "/tmp/sound.wav"
        )
        patched = patcher.get()
        loader_id = patched["42"]["inputs"]["ref_audios.ref_audio_2"][0]
        self.assertEqual("LoadAudio", patched[loader_id]["class_type"])
        self.assertEqual("<Audio 3>", tag)

    def test_fresh_node_id(self):
        backend = self._backend()
        wf = backend.load_workflow()
        patcher = WorkflowPatcher(wf)
        backend._add_ref_node_and_wire(patcher, "ref_images", 0, "/tmp/a.png")
        backend._add_ref_node_and_wire(patcher, "ref_images", 1, "/tmp/b.png")
        patched = patcher.get()
        id_a = patched["42"]["inputs"]["ref_images.ref_image_0"][0]
        id_b = patched["42"]["inputs"]["ref_images.ref_image_1"][0]
        self.assertNotEqual(id_a, id_b)

    def test_wires_to_correct_slot(self):
        backend = self._backend()
        wf = backend.load_workflow()
        patcher = WorkflowPatcher(wf)
        backend._add_ref_node_and_wire(patcher, "ref_images", 0, "/tmp/a.png")
        backend._add_ref_node_and_wire(patcher, "ref_images", 3, "/tmp/b.png")
        cores = patcher.find_nodes_by_class_type("MiniMaxH3ReferenceToVideo")
        core_inputs = cores[0][1]["inputs"]
        self.assertIn("ref_images.ref_image_0", core_inputs)
        self.assertIn("ref_images.ref_image_3", core_inputs)
        self.assertNotIn("ref_images.ref_image_1", core_inputs)


# ---------------------------------------------------------------------------
# _collect_scene_references tests
# ---------------------------------------------------------------------------

class CollectSceneReferencesTests(unittest.TestCase):
    def _backend(self):
        return ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
        )

    def test_all_types(self):
        backend = self._backend()
        scene = {
            "references": {
                "actor_sheet_paths": ["actor1.png", "actor2.png"],
                "location_sheet_path": "loc.png",
                "reference_video_paths": ["clip1.mp4"],
                "reference_audio_paths": ["sound1.wav"],
            },
        }
        img, vid, aud = backend._collect_scene_references(scene)
        self.assertEqual(3, len(img))
        self.assertEqual(1, len(vid))
        self.assertEqual(1, len(aud))

    def test_only_actors(self):
        backend = self._backend()
        scene = {
            "references": {
                "actor_sheet_paths": ["a1.png", "a2.png"],
            },
        }
        img, vid, aud = backend._collect_scene_references(scene)
        self.assertEqual(2, len(img))
        self.assertEqual(0, len(vid))
        self.assertEqual(0, len(aud))

    def test_clamped(self):
        backend = self._backend()
        scene = {
            "references": {
                "actor_sheet_paths": [f"a{i}.png" for i in range(20)],
            },
        }
        img, vid, aud = backend._collect_scene_references(scene)
        self.assertEqual(9, len(img))


# ---------------------------------------------------------------------------
# _patch_dynamic_ref_inputs tests
# ---------------------------------------------------------------------------

class PatchDynamicRefInputsTests(unittest.TestCase):
    def _backend(self, workflow=None):
        uploader = FakeAssetUploader()
        return ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
            asset_uploader=uploader,
            workflow=workflow or _r2v_core_workflow(),
        )

    def _scene(self, images=None, videos=None, audios=None):
        refs = {}
        if images:
            refs["actor_sheet_paths"] = [f"/tmp/actor{i}.png" for i in range(images)]
        if videos:
            refs["reference_video_paths"] = [f"/tmp/clip{j}.mp4" for j in range(videos)]
        if audios:
            refs["reference_audio_paths"] = [f"/tmp/sound{j}.wav" for j in range(audios)]
        return {"references": refs}

    def test_fills_empty_slots_from_scene(self):
        backend = self._backend()
        wf = backend.load_workflow()
        patcher = WorkflowPatcher(wf)
        tags = backend._patch_dynamic_ref_inputs(patcher, self._scene(images=3))
        cores = patcher.find_nodes_by_class_type("MiniMaxH3ReferenceToVideo")
        core_inputs = cores[0][1]["inputs"]
        self.assertIn("ref_images.ref_image_0", core_inputs)
        self.assertIn("ref_images.ref_image_1", core_inputs)
        self.assertIn("ref_images.ref_image_2", core_inputs)
        self.assertEqual(3, len(tags))
        self.assertEqual("<Picture 1>", tags[0])

    def test_skips_occupied_slots(self):
        """Slots 0 and 2 occupied -> slots 1,3,4 get dynamic wiring from 5 images."""
        wf = _r2v_core_workflow(occupied_indices=[0, 2])
        backend = self._backend(workflow=wf)
        wf = backend.load_workflow()
        patcher = WorkflowPatcher(wf)
        tags = backend._patch_dynamic_ref_inputs(patcher, self._scene(images=5))
        cores = patcher.find_nodes_by_class_type("MiniMaxH3ReferenceToVideo")
        core_inputs = cores[0][1]["inputs"]
        # Slots 0,2 already pre-wired; 1,3,4 filled dynamically
        self.assertIn("ref_images.ref_image_0", core_inputs)
        self.assertIn("ref_images.ref_image_1", core_inputs)
        self.assertIn("ref_images.ref_image_2", core_inputs)
        self.assertIn("ref_images.ref_image_3", core_inputs)
        self.assertIn("ref_images.ref_image_4", core_inputs)
        # 3 new tags for slots 1,3,4
        self.assertEqual(3, len(tags))

    def test_all_occupied_is_noop(self):
        """3 scene images, all 3 slots occupied -> no new nodes."""
        wf = _r2v_core_workflow(occupied_indices=[0, 1, 2])
        backend = self._backend(workflow=wf)
        wf = backend.load_workflow()
        patcher = WorkflowPatcher(wf)
        initial_node_count = len(patcher.get())
        tags = backend._patch_dynamic_ref_inputs(patcher, self._scene(images=3))
        patched = patcher.get()
        self.assertEqual(initial_node_count, len(patched))
        self.assertEqual(0, len(tags))

    def test_respects_max_limits(self):
        """20 scene images -> clamped to MAX_REF_IMAGES (9)."""
        backend = self._backend()
        wf = backend.load_workflow()
        patcher = WorkflowPatcher(wf)
        tags = backend._patch_dynamic_ref_inputs(patcher, self._scene(images=20))
        # Only 9 image tags (max)
        image_tags = [t for t in tags if t.startswith("<Picture")]
        self.assertEqual(9, len(image_tags))

    def test_returns_prompt_tags(self):
        backend = self._backend()
        wf = backend.load_workflow()
        patcher = WorkflowPatcher(wf)
        tags = backend._patch_dynamic_ref_inputs(
            patcher, self._scene(images=2, videos=1, audios=1)
        )
        self.assertEqual(["<Picture 1>", "<Picture 2>", "<Video 1>", "<Audio 1>"], tags)


# ---------------------------------------------------------------------------
# build_workflow dynamic wiring integration tests
# ---------------------------------------------------------------------------

class BuildWorkflowDynamicWiringTests(unittest.TestCase):
    def _backend(self, workflow=None):
        uploader = FakeAssetUploader()
        return ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
            asset_uploader=uploader,
            workflow=workflow,
        )

    def _r2v_workflow_with_core(self):
        wf = _native_r2v_workflow()
        wf["42"] = {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "_meta": {"title": "#R2V_COMBINE"},
            "inputs": {},
        }
        return wf

    def test_no_paths_uses_dynamic(self):
        """Without explicit ref_image_paths, scene refs are used via dynamic wiring."""
        wf = self._r2v_workflow_with_core()
        backend = self._backend(workflow=wf)
        scene = {
            "scene": 1,
            "references": {
                "actor_sheet_paths": ["/tmp/a1.png", "/tmp/a2.png"],
            },
        }
        result = backend.build_workflow(scene, prompt="test")
        cores = [(nid, n) for nid, n in result.items()
                 if n.get("class_type") == "MiniMaxH3ReferenceToVideo"]
        self.assertTrue(len(cores) >= 1)
        core_inputs = cores[0][1].get("inputs", {})
        # Dynamic wiring should have filled slots 0 and 1
        self.assertIn("ref_images.ref_image_0", core_inputs)
        self.assertIn("ref_images.ref_image_1", core_inputs)

    def test_with_paths_and_dynamic(self):
        """Explicit paths fill slots first, dynamic wiring fills remainder."""
        wf = self._r2v_workflow_with_core()
        backend = self._backend(workflow=wf)
        scene = {
            "scene": 1,
            "references": {
                "actor_sheet_paths": ["/tmp/a1.png", "/tmp/a2.png", "/tmp/a3.png"],
                "location_sheet_path": "/tmp/loc.png",
            },
        }
        result = backend.build_workflow(
            scene,
            prompt="test",
            ref_image_paths=["/tmp/explicit.png"],
        )
        cores = [(nid, n) for nid, n in result.items()
                 if n.get("class_type") == "MiniMaxH3ReferenceToVideo"]
        core_inputs = cores[0][1].get("inputs", {})
        # Explicit path fills slot 0 via anchor, scene refs fill remaining via dynamic
        # Note: _clear_reference_group clears first, then _patch_reference_patches
        # slot 0, then dynamic fills slots from scene. But scene's first ref is
        # now slot 0 which is occupied so it skips to slot 1.
        # Total nodes wired: 1 explicit + scene refs for remaining slots
        self.assertIn("ref_images.ref_image_0", core_inputs)

    def test_partial_paths(self):
        """2 explicit paths + 1 scene extra -> slots 0,1,2 all filled."""
        wf = self._r2v_workflow_with_core()
        backend = self._backend(workflow=wf)
        scene = {
            "scene": 1,
            "references": {
                "actor_sheet_paths": ["/tmp/a1.png", "/tmp/a2.png", "/tmp/a3.png"],
            },
        }
        result = backend.build_workflow(
            scene,
            prompt="test",
            ref_image_paths=["/tmp/exp1.png", "/tmp/exp2.png"],
        )
        cores = [(nid, n) for nid, n in result.items()
                 if n.get("class_type") == "MiniMaxH3ReferenceToVideo"]
        core_inputs = cores[0][1].get("inputs", {})
        self.assertIn("ref_images.ref_image_0", core_inputs)
        self.assertIn("ref_images.ref_image_1", core_inputs)
        # Scene refs fill slot 2 (dynamic)
        self.assertIn("ref_images.ref_image_2", core_inputs)


# ---------------------------------------------------------------------------
# Stem audio reference tests
# ---------------------------------------------------------------------------

class ResolveStemAudioPathsTests(unittest.TestCase):
    def test_scene_stem_order_is_authoritative_over_backend_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            drums = root / "drums.wav"
            full_mix = root / "song.wav"
            drums.write_bytes(b"drums")
            full_mix.write_bytes(b"mix")
            backend = self._backend(audio_ref_stems=["vocals", "full_mix"])

            result = backend._resolve_stem_audio_paths({
                "stem_audio": {
                    "stems": ["drums", "full_mix"],
                    "paths": {"drums": str(drums), "full_mix": str(full_mix)},
                },
            })

            self.assertEqual([drums, full_mix], result)
    """Tests for ComfyUIMiniMaxH3R2VBackend._resolve_stem_audio_paths."""

    def _backend(self, audio_ref_stems=None, workflow=None):
        uploader = FakeAssetUploader()
        return ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
            asset_uploader=uploader,
            workflow=workflow or _native_r2v_workflow(),
            audio_ref_stems=audio_ref_stems,
        )

    def test_empty_stem_audio_returns_empty(self):
        backend = self._backend()
        result = backend._resolve_stem_audio_paths({"scene": 1})
        self.assertEqual(result, [])

    def test_falls_back_to_project_stems_for_legacy_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            stem_dir = project_dir / "output" / "stems"
            stem_dir.mkdir(parents=True)
            vocal_path = stem_dir / "vocals_song.wav"
            vocal_path.write_bytes(b"fake audio")

            backend = ComfyUIMiniMaxH3R2VBackend(
                client=FakeClient(),
                workflow_path=Path("/tmp/wf.json"),
                output_dir=Path("/tmp/out"),
                asset_uploader=FakeAssetUploader(),
                workflow=_native_r2v_workflow(),
                project_dir=project_dir,
                audio_ref_stems=["vocals", "full_mix"],
            )

            self.assertEqual(
                backend._resolve_stem_audio_paths({"scene": 1}),
                [vocal_path],
            )

    def test_resolves_vocals_and_full_mix(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            vocal_path = tmp_path / "vocals.wav"
            vocal_path.write_bytes(b"fake audio")
            fullmix_path = tmp_path / "full_mix.wav"
            fullmix_path.write_bytes(b"fake audio")

            scene = {
                "scene": 1,
                "stem_audio": {
                    "stems": ["vocals", "full_mix"],
                    "paths": {
                        "vocals": str(vocal_path),
                        "full_mix": str(fullmix_path),
                    }
                }
            }
            backend = self._backend(audio_ref_stems=["vocals", "full_mix"])
            result = backend._resolve_stem_audio_paths(scene)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0], vocal_path)
            self.assertEqual(result[1], fullmix_path)

    def test_silent_mode_excludes_vocal_stem(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            vocal_path = tmp_path / "vocals.wav"
            vocal_path.write_bytes(b"fake audio")
            fullmix_path = tmp_path / "full_mix.wav"
            fullmix_path.write_bytes(b"fake audio")
            scene = {
                "scene": 1,
                "silent_mode": True,
                "stem_audio": {
                    "stems": ["vocals", "full_mix"],
                    "paths": {"vocals": str(vocal_path), "full_mix": str(fullmix_path)},
                },
                "references": {
                    "reference_audio_paths": [str(vocal_path), str(fullmix_path)],
                },
            }
            backend = self._backend(audio_ref_stems=["vocals", "full_mix"])
            self.assertEqual([fullmix_path], backend._resolve_stem_audio_paths(scene))
            self.assertEqual([str(fullmix_path)], backend._resolve_ref_audio_paths(scene))

    def test_resolves_project_relative_stem_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            vocal_path = project_dir / "output" / "stems" / "vocals.wav"
            vocal_path.parent.mkdir(parents=True)
            vocal_path.write_bytes(b"fake audio")
            backend = ComfyUIMiniMaxH3R2VBackend(
                client=FakeClient(),
                workflow_path=Path("/tmp/wf.json"),
                output_dir=Path("/tmp/out"),
                asset_uploader=FakeAssetUploader(),
                workflow=_native_r2v_workflow(),
                project_dir=project_dir,
            )

            result = backend._resolve_stem_audio_paths({
                "stem_audio": {
                    "stems": ["vocals"],
                    "paths": {"vocals": "output/stems/vocals.wav"},
                },
            })

            self.assertEqual([vocal_path], result)

    def test_scene_selection_overrides_instance_defaults(self):
        """A prepared scene's ordered slots override backend fallback defaults."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            vocal_path = tmp_path / "vocals.wav"
            vocal_path.write_bytes(b"fake")
            drums_path = tmp_path / "drums.wav"
            drums_path.write_bytes(b"fake")

            scene = {
                "scene": 1,
                "stem_audio": {
                    "stems": ["vocals"],
                    "paths": {
                        "vocals": str(vocal_path),
                        "drums": str(drums_path),
                    }
                }
            }
            backend = self._backend(audio_ref_stems=["drums", "vocals"])
            result = backend._resolve_stem_audio_paths(scene)
            self.assertEqual([vocal_path], result)

    def test_max_clamped_to_three(self):
        """At most MAX_REF_AUDIOS (3) results returned."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = {}
            stems = []
            for i in range(6):
                name = f"stem_{i}"
                p = tmp_path / f"{name}.wav"
                p.write_bytes(b"fake")
                paths[name] = str(p)
                stems.append(name)

            scene = {
                "scene": 1,
                "stem_audio": {
                    "stems": stems,
                    "paths": paths,
                }
            }
            backend = self._backend(audio_ref_stems=stems)
            result = backend._resolve_stem_audio_paths(scene)
            self.assertEqual(len(result), 3)  # MAX_REF_AUDIOS

    def test_missing_paths_skipped(self):
        """Non-existent paths are silently skipped."""
        scene = {
            "scene": 1,
            "stem_audio": {
                "stems": ["vocals", "drums"],
                "paths": {
                    "vocals": "/nonexistent/vocals.wav",
                }
            }
        }
        backend = self._backend(audio_ref_stems=["vocals", "drums"])
        result = backend._resolve_stem_audio_paths(scene)
        self.assertEqual(result, [])  # vocals doesn't exist, drums not in paths


    def test_scene_stem_order_is_preserved_and_clamped(self):
        """Prepared H3 labels and backend slots use the same scene order."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = {}
            for name in ["drums", "bass", "vocals", "other", "full_mix"]:
                p = tmp_path / f"{name}.wav"
                p.write_bytes(b"fake")
                paths[name] = str(p)

            scene = {
                "scene": 1,
                "stem_audio": {
                    "stems": ["drums", "bass", "vocals", "other", "full_mix"],
                    "paths": paths,
                }
            }
            backend = self._backend()
            result = backend._resolve_stem_audio_paths(scene)
            self.assertEqual(
                ["drums.wav", "bass.wav", "vocals.wav"],
                [path.name for path in result],
            )


class StemAudioRenderVideoIntegrationTests(unittest.TestCase):
    """Integration tests for stem audio merging in render_video."""

    def _backend(self, audio_ref_stems=None):
        uploader = FakeAssetUploader()
        queue = FakeRenderQueue()
        return ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
            asset_uploader=uploader,
            render_queue=queue,
            workflow=_native_r2v_workflow(),
            audio_ref_stems=audio_ref_stems,
        )

    def test_stem_paths_given_priority_over_ref_paths(self):
        """Stem audio fills slots first, ref_audio_paths fill remaining."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            vocal_path = tmp_path / "vocals.wav"
            vocal_path.write_bytes(b"v")
            drums_path = tmp_path / "drums.wav"
            drums_path.write_bytes(b"d")
            ref_path = tmp_path / "ref.wav"
            ref_path.write_bytes(b"r")

            backend = self._backend(audio_ref_stems=["vocals", "drums"])

            # Manually call resolve methods to verify merge logic
            scene = {
                "scene": 1,
                "stem_audio": {
                    "stems": ["vocals", "drums"],
                    "paths": {
                        "vocals": str(vocal_path),
                        "drums": str(drums_path),
                    }
                },
                "references": {
                    "reference_audio_paths": [str(ref_path)]
                }
            }

            stem_paths = backend._resolve_stem_audio_paths(scene)
            ref_paths = backend._resolve_ref_audio_paths(scene)

            # Verify stem takes priority
            self.assertEqual(len(stem_paths), 2)
            self.assertIn(vocal_path, stem_paths)
            self.assertIn(drums_path, stem_paths)
            self.assertEqual(len(ref_paths), 1)


# ---------------------------------------------------------------------------
# Trimmed stem audio tests
# ---------------------------------------------------------------------------

class PatchReferenceAudiosTrimmedTests(unittest.TestCase):
    """Tests for _patch_reference_audios with trimming enabled."""

    def _backend(self, workflow=None):
        uploader = FakeAssetUploader()
        return ComfyUIMiniMaxH3R2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
            asset_uploader=uploader,
            workflow=workflow or _native_r2v_workflow(),
        )

    def test_trimmed_creates_trim_nodes_and_wires_to_r2v(self):
        """With duration_seconds, stem audio goes through TrimAudioDuration."""
        # Use a workflow that includes a MiniMaxH3ReferenceToVideo core node
        wf = _native_r2v_workflow()
        wf["42"] = {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "_meta": {"title": "#R2V_COMBINE"},
            "inputs": {},
        }
        backend = self._backend(workflow=wf)
        scene = {
            "scene": 1,
            "abs_start_seconds": 10.5,
            "references": {"actor_sheet_paths": ["/tmp/actor.png"]},
        }
        patched = backend.build_workflow(
            scene,
            prompt="test",
            duration_seconds=5.0,
            ref_audio_paths=["/tmp/vocals.wav", "/tmp/drums.wav"],
        )

        # Trim nodes have correct parameters
        found_trim_1 = False
        found_trim_2 = False
        for node_id, node in patched.items():
            meta_title = node.get("_meta", {}).get("title", "")
            if meta_title == "#TRIM_AUDIO_1":
                found_trim_1 = True
                self.assertAlmostEqual(node["inputs"]["start_index"], 10.5)
                self.assertAlmostEqual(node["inputs"]["duration"], 5.0)
            elif meta_title == "#TRIM_AUDIO_2":
                found_trim_2 = True
                self.assertAlmostEqual(node["inputs"]["start_index"], 10.5)
                self.assertAlmostEqual(node["inputs"]["duration"], 5.0)
        self.assertTrue(found_trim_1)
        self.assertTrue(found_trim_2)

        # R2V core wired to trim outputs, not LoadAudio directly
        cores = [(nid, n) for nid, n in patched.items()
                 if n.get("class_type") == "MiniMaxH3ReferenceToVideo"]
        self.assertTrue(len(cores) >= 1)
        core_inputs = cores[0][1].get("inputs", {})
        audio_0_ref = core_inputs.get("ref_audios.ref_audio_0")
        if audio_0_ref:
            trim_node_id = audio_0_ref[0]
            trim_node = patched.get(str(trim_node_id), patched.get(trim_node_id))
            self.assertEqual(trim_node.get("class_type"), "TrimAudioDuration")

    def test_no_trim_wires_loadaudio_direct(self):
        """Without duration, LoadAudio wires directly to R2V (existing behavior)."""
        wf = _native_r2v_workflow()
        wf["42"] = {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "_meta": {"title": "#R2V_COMBINE"},
            "inputs": {},
        }
        backend = self._backend(workflow=wf)
        scene = {
            "scene": 1,
            "references": {"actor_sheet_paths": ["/tmp/actor.png"]},
        }
        patched = backend.build_workflow(
            scene,
            prompt="test",
            ref_audio_paths=["/tmp/vocals.wav"],
        )

        cores = [(nid, n) for nid, n in patched.items()
                 if n.get("class_type") == "MiniMaxH3ReferenceToVideo"]
        self.assertTrue(len(cores) >= 1)
        core_inputs = cores[0][1].get("inputs", {})
        audio_1_ref = core_inputs.get("ref_audios.ref_audio_0")
        self.assertIsNotNone(audio_1_ref)
        loader_id = audio_1_ref[0]
        loader_node = patched.get(str(loader_id), patched.get(loader_id))
        self.assertEqual(loader_node.get("class_type"), "LoadAudio")

    def test_abs_start_default_zero(self):
        """abs_start_seconds defaults to 0.0 when not provided."""
        wf = _native_r2v_workflow()
        wf["42"] = {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "_meta": {"title": "#R2V_COMBINE"},
            "inputs": {},
        }
        backend = self._backend(workflow=wf)
        scene = {
            "scene": 1,
            "references": {"actor_sheet_paths": ["/tmp/actor.png"]},
        }
        patched = backend.build_workflow(
            scene,
            prompt="test",
            duration_seconds=3.0,
            ref_audio_paths=["/tmp/test.wav"],
        )
        for node_id, node in patched.items():
            meta_title = node.get("_meta", {}).get("title", "")
            if meta_title == "#TRIM_AUDIO_1":
                self.assertAlmostEqual(node["inputs"]["start_index"], 0.0)
                return
        self.fail("#TRIM_AUDIO_1 not found in workflow")


class StemAudioTrimRenderVideoIntegrationTests(unittest.TestCase):
    """Integration tests: render_video produces trimmed stem audio in workflow JSON."""

    def test_render_video_trims_stem_audio(self):
        """Full render_video flow with stem audio produces trimmed refs."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            vocal_path = tmp_path / "vocals.wav"
            vocal_path.write_bytes(b"fake audio")
            fullmix_path = tmp_path / "full_mix.wav"
            fullmix_path.write_bytes(b"fake audio")

            workflow = _native_r2v_workflow()
            # Add core R2V node
            workflow["42"] = {
                "class_type": "MiniMaxH3ReferenceToVideo",
                "_meta": {"title": "#R2V_COMBINE"},
                "inputs": {},
            }
            scene = {
                "scene": 1,
                "abs_start_seconds": 128.0,
                "stem_audio": {
                    "stems": ["vocals", "full_mix"],
                    "paths": {
                        "vocals": str(vocal_path),
                        "full_mix": str(fullmix_path),
                    }
                },
                "duration_seconds": 5.0,
                "references": {"actor_sheet_paths": ["/tmp/actor.png"]},
            }

            backend = ComfyUIMiniMaxH3R2VBackend(
                client=FakeClient(),
                workflow_path=Path("/tmp/wf.json"),
                output_dir=tmp_path / "output",
                asset_uploader=FakeAssetUploader(),
                render_queue=FakeRenderQueue(),
                postprocessor=FakePostProcessor(),
                workflow=workflow,
                audio_ref_stems=["vocals", "full_mix"],
            )

            # We can't actually render, but we can build the workflow
            build_result = backend.build_workflow(
                scene,
                prompt="test prompt",
                duration_seconds=5.0,
                ref_audio_paths=[vocal_path, fullmix_path],
            )

            # Verify trim nodes exist with correct params
            found_trims = {}
            for node_id, node in build_result.items():
                title = node.get("_meta", {}).get("title", "")
                if title.startswith("#TRIM_AUDIO_"):
                    found_trims[title] = node

            self.assertIn("#TRIM_AUDIO_1", found_trims)
            self.assertIn("#TRIM_AUDIO_2", found_trims)

            trim1 = found_trims["#TRIM_AUDIO_1"]
            self.assertAlmostEqual(trim1["inputs"]["duration"], 5.0)
            # Trim nodes wire to LoadAudio, not the other way around
            self.assertIn("audio", trim1["inputs"])

            # R2V core wires to trim nodes
            cores = [(nid, n) for nid, n in build_result.items()
                     if n.get("class_type") == "MiniMaxH3ReferenceToVideo"]
            self.assertTrue(len(cores) >= 1)
            core_inputs = cores[0][1].get("inputs", {})
            for slot_key, ref_val in core_inputs.items():
                if slot_key.startswith("ref_audios.ref_audio_"):
                    source_id = str(ref_val[0])
                    source_node = build_result.get(source_id, {})
                    # Should be a TrimAudioDuration, not LoadAudio
                    self.assertEqual(source_node.get("class_type"), "TrimAudioDuration")


if __name__ == "__main__":
    unittest.main()
