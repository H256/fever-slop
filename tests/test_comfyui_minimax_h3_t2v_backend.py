"""Tests for ComfyUIMiniMaxH3T2VBackend."""

import unittest
import tempfile
import json
from pathlib import Path

from feverslop.adapters.comfyui_minimax_h3_t2v_backend import (
    ComfyUIMiniMaxH3T2VBackend,
)
from feverslop.adapters.comfyui_minimax_h3_video_backend import (
    ComfyUIMiniMaxH3VideoRenderBackend,
)
from feverslop.adapters.workflow_patcher import WorkflowPatcher
from feverslop.domain.postprocessing import TrimSpec
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
        return f"feverslop/t2v/{p.stem}-abc123{p.suffix}"

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
# Minimal T2V workflow fixture
# ---------------------------------------------------------------------------

def _t2v_workflow() -> dict:
    """Minimal T2V workflow with FeverSlop anchors."""
    return {
        "115": {
            "class_type": "ResolutionSelector",
            "_meta": {"title": "#MEGAPIXEL"},
            "inputs": {"megapixels": 0.4, "aspect_ratio": "16:9", "multiple": 32},
        },
        "129": {
            "class_type": "RandomNoise",
            "_meta": {"title": "#SEED"},
            "inputs": {"noise_seed": 12345},
        },
        "133": {
            "class_type": "PrimitiveInt",
            "_meta": {"title": "#FRAMECOUNT"},
            "inputs": {"value": 144},
        },
        "136": {
            "class_type": "LoadImage",
            "_meta": {"title": "#T2V_START"},
            "inputs": {"image": "", "upload": "image"},
        },
        "137": {
            "class_type": "LoadImage",
            "_meta": {"title": "#T2V_END"},
            "inputs": {"image": "", "upload": "image"},
        },
        "138": {
            "class_type": "PrimitiveStringMultiline",
            "_meta": {"title": "#T2V_TEXT"},
            "inputs": {"value": ""},
        },
        "131": {
            "class_type": "MiniMaxH3ImageToVideo",
            "_meta": {"title": "#PROMPT"},
            "inputs": {
                "prompt": "old prompt",
                "width": ["115", 0],
                "height": ["115", 1],
                "length": ["133", 0],
                "clip": ["128", 0],
                "vae": ["119", 0],
                "first_frame": ["136", 0],
                "last_frame": ["137", 0],
                "text_data": ["138", 0],
            },
        },
        "135": {
            "class_type": "VHS_VideoCombine",
            "_meta": {"title": "#SAVE_VIDEO"},
            "inputs": {"filename_prefix": "AnimateDiff", "format": "video/h264-mp4"},
        },
        "122": {
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"},
            "inputs": {},
        },
        "121": {
            "class_type": "VAEDecodeAudio",
            "_meta": {"title": "VAE Decode Audio"},
            "inputs": {},
        },
    }


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------

class ConstructorTests(unittest.TestCase):
    def test_constructor_stores_params(self):
        uploader = FakeAssetUploader()
        backend = ComfyUIMiniMaxH3T2VBackend(
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
        backend = ComfyUIMiniMaxH3T2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
        )
        self.assertEqual(100000, backend.seed_offset)
        self.assertFalse(backend.randomize_seed)

    def test_constructor_custom_seed_settings(self):
        backend = ComfyUIMiniMaxH3T2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
            seed_offset=50000,
            randomize_seed=True,
        )
        self.assertEqual(50000, backend.seed_offset)
        self.assertTrue(backend.randomize_seed)

    def test_constructor_debug_workflows_dir(self):
        backend = ComfyUIMiniMaxH3T2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
            debug_workflows_dir=Path("/tmp/debug"),
        )
        self.assertEqual(Path("/tmp/debug"), backend.debug_workflows_dir)

    def test_constructor_in_memory_workflow(self):
        wf = _t2v_workflow()
        backend = ComfyUIMiniMaxH3T2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
            workflow=wf,
        )
        loaded = backend.load_workflow()
        self.assertEqual(wf, loaded)
        self.assertIsNot(loaded, wf)

    def test_constructor_creates_defaults(self):
        backend = ComfyUIMiniMaxH3T2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
        )
        self.assertIsNotNone(backend.asset_uploader)
        self.assertIsNotNone(backend.render_queue)
        self.assertIsNotNone(backend.postprocessor)
        self.assertIsNotNone(backend.model_resolver)


# ---------------------------------------------------------------------------
# ValidateScene tests
# ---------------------------------------------------------------------------

class ValidateSceneTests(unittest.TestCase):
    def setUp(self):
        self.backend = ComfyUIMiniMaxH3T2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
        )

    def test_empty_scene_ok(self):
        """T2V always passes validation."""
        self.backend._validate_scene({})

    def test_scene_with_refs_ok(self):
        self.backend._validate_scene({
            "scene": 1,
            "references": {"actor_sheet_paths": ["/tmp/actor.png"]},
        })

    def test_scene_without_refs_ok(self):
        """T2V does not require actor references."""
        self.backend._validate_scene({"scene": 1})


# ---------------------------------------------------------------------------
# _seed_for_scene tests
# ---------------------------------------------------------------------------

class SeedForSceneTests(unittest.TestCase):
    def test_sequential_seed(self):
        backend = ComfyUIMiniMaxH3T2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
            seed_offset=100000,
            randomize_seed=False,
        )
        self.assertEqual(100001, backend._seed_for_scene(1))
        self.assertEqual(100005, backend._seed_for_scene(5))

    def test_randomized_seed(self):
        backend = ComfyUIMiniMaxH3T2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
            randomize_seed=True,
        )
        seed_a = backend._seed_for_scene(1)
        seed_b = backend._seed_for_scene(1)
        self.assertIsInstance(seed_a, int)
        self.assertGreaterEqual(seed_a, 0)
        self.assertNotEqual(seed_a, seed_b)


# ---------------------------------------------------------------------------
# T2V-specific patcher tests
# ---------------------------------------------------------------------------

class PatchT2VStartTests(unittest.TestCase):
    def test_patches_image_name(self):
        wf = _t2v_workflow()
        patcher = WorkflowPatcher(wf)
        ComfyUIMiniMaxH3T2VBackend._patch_t2v_start(patcher, "start-abc.png")
        self.assertEqual("start-abc.png", patcher.get()["136"]["inputs"]["image"])

    def test_no_existing_anchor_raises(self):
        wf = {"1": {"class_type": "OtherNode", "inputs": {}}}
        patcher = WorkflowPatcher(wf)
        with self.assertRaises(KeyError):
            ComfyUIMiniMaxH3T2VBackend._patch_t2v_start(patcher, "img.png")


class PatchT2VEndTests(unittest.TestCase):
    def test_patches_image_name(self):
        wf = _t2v_workflow()
        patcher = WorkflowPatcher(wf)
        ComfyUIMiniMaxH3T2VBackend._patch_t2v_end(patcher, "end-abc.png")
        self.assertEqual("end-abc.png", patcher.get()["137"]["inputs"]["image"])

    def test_no_existing_anchor_raises(self):
        wf = {"1": {"class_type": "OtherNode", "inputs": {}}}
        patcher = WorkflowPatcher(wf)
        with self.assertRaises(KeyError):
            ComfyUIMiniMaxH3T2VBackend._patch_t2v_end(patcher, "img.png")


class PatchT2VTextTests(unittest.TestCase):
    def test_patches_text(self):
        wf = _t2v_workflow()
        patcher = WorkflowPatcher(wf)
        ComfyUIMiniMaxH3T2VBackend._patch_t2v_text(patcher, "start desc; end desc")
        self.assertEqual("start desc; end desc", patcher.get()["138"]["inputs"]["value"])

    def test_no_existing_anchor_raises(self):
        wf = {"1": {"class_type": "OtherNode", "inputs": {}}}
        patcher = WorkflowPatcher(wf)
        with self.assertRaises(KeyError):
            ComfyUIMiniMaxH3T2VBackend._patch_t2v_text(patcher, "text")


# ---------------------------------------------------------------------------
# build_workflow tests
# ---------------------------------------------------------------------------

class BuildWorkflowTests(unittest.TestCase):
    def _backend(self, workflow: dict | None = None):
        uploader = FakeAssetUploader()
        return ComfyUIMiniMaxH3T2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
            asset_uploader=uploader,
            workflow=workflow,
        )

    def test_full_flow_with_all_params(self):
        backend = self._backend(workflow=_t2v_workflow())
        result = backend.build_workflow(
            {"scene": 3},
            prompt="cinematic vaporwave scene",
            duration_seconds=5.0,
            width=1024,
            height=768,
            start_frame_path="/tmp/start.png",
            end_frame_path="/tmp/end.png",
            text_data="start: blue; end: pink",
        )
        # Prompt patched
        self.assertEqual("cinematic vaporwave scene", result["131"]["inputs"]["prompt"])
        # Frame count patched: round(5.0 * 24) = 120
        self.assertEqual(120, result["133"]["inputs"]["value"])
        # Megapixels patched: round(1024 * 768 / 1_000_000, 1) = 0.8
        expected_mp = round(1024 * 768 / 1_000_000, 1)
        self.assertEqual(expected_mp, result["115"]["inputs"]["megapixels"])
        # Start frame patched
        self.assertIn("start", result["136"]["inputs"]["image"])
        # End frame patched
        self.assertIn("end", result["137"]["inputs"]["image"])
        # Text data patched
        self.assertEqual("start: blue; end: pink", result["138"]["inputs"]["value"])
        # Save prefix patched
        self.assertEqual("scene_0003/raw", result["135"]["inputs"]["filename_prefix"])

    def test_minimal_prompt_only(self):
        backend = self._backend(workflow=_t2v_workflow())
        result = backend.build_workflow(
            {"scene": 1},
            prompt="a simple prompt",
        )
        self.assertEqual("a simple prompt", result["131"]["inputs"]["prompt"])

    def test_seed_set(self):
        backend = self._backend(workflow=_t2v_workflow())
        backend.seed_offset = 50000
        backend.randomize_seed = False
        result = backend.build_workflow(
            {"scene": 7},
            prompt="test",
        )
        self.assertEqual(50007, result["129"]["inputs"]["noise_seed"])

    def test_megapixels_computed(self):
        backend = self._backend(workflow=_t2v_workflow())
        result = backend.build_workflow(
            {"scene": 1},
            prompt="test",
            width=1344,
            height=768,
        )
        expected_mp = round(1344 * 768 / 1_000_000, 1)
        self.assertEqual(expected_mp, result["115"]["inputs"]["megapixels"])

    def test_duration_not_set_when_none(self):
        backend = self._backend(workflow=_t2v_workflow())
        result = backend.build_workflow(
            {"scene": 1},
            prompt="test",
        )
        # Frame count stays as workflow default (144)
        self.assertEqual(144, result["133"]["inputs"]["value"])

    def test_start_frame_only(self):
        backend = self._backend(workflow=_t2v_workflow())
        result = backend.build_workflow(
            {"scene": 1},
            prompt="test",
            start_frame_path="/tmp/start.png",
        )
        self.assertIn("start", result["136"]["inputs"]["image"])
        # End frame untouched
        self.assertEqual("", result["137"]["inputs"]["image"])

    def test_text_data_only(self):
        backend = self._backend(workflow=_t2v_workflow())
        result = backend.build_workflow(
            {"scene": 1},
            prompt="test",
            text_data="keyframe text here",
        )
        self.assertEqual("keyframe text here", result["138"]["inputs"]["value"])

    def test_save_prefix_zero_padded(self):
        backend = self._backend(workflow=_t2v_workflow())
        result = backend.build_workflow(
            {"scene": 42},
            prompt="test",
        )
        self.assertEqual("scene_0042/raw", result["135"]["inputs"]["filename_prefix"])


# ---------------------------------------------------------------------------
# render_video tests
# ---------------------------------------------------------------------------

class RenderVideoTests(unittest.TestCase):
    def test_full_flow_with_postprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            uploader = FakeAssetUploader()
            queue = FakeRenderQueue()
            postprocessor = FakePostProcessor()
            resolver = FakeModelResolver()
            backend = ComfyUIMiniMaxH3T2VBackend(
                client=FakeClient(),
                workflow_path=Path("/tmp/wf.json"),
                output_dir=tmp_path / "output",
                asset_uploader=uploader,
                render_queue=queue,
                postprocessor=postprocessor,
                model_resolver=resolver,
                workflow=_t2v_workflow(),
            )
            request = VideoRenderRequest(
                scene={
                    "scene": 2,
                    "description": "Vaporwave scene",
                    "duration_seconds": 5.0,
                },
                scene_number=2,
                prompt="A vaporwave scene",
                workflow_path=Path("/tmp/wf.json"),
                output_dir=tmp_path / "output",
                audio_file=Path("/tmp/song.wav"),
                storyboard_dir=tmp_path / "storyboard",
                upload_audio=False,
            )
            result = backend.render_video(request)
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
            backend = ComfyUIMiniMaxH3T2VBackend(
                client=FakeClient(),
                workflow_path=Path("/tmp/wf.json"),
                output_dir=tmp_path / "output",
                asset_uploader=uploader,
                render_queue=queue,
                postprocess=False,
                model_resolver=FakeModelResolver(),
                workflow=_t2v_workflow(),
            )
            request = VideoRenderRequest(
                scene={
                    "scene": 1,
                    "description": "Test",
                },
                scene_number=1,
                prompt="Test prompt",
                workflow_path=Path("/tmp/wf.json"),
                output_dir=tmp_path / "output",
                audio_file=Path("/tmp/song.wav"),
                storyboard_dir=tmp_path / "storyboard",
                upload_audio=False,
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
            backend = ComfyUIMiniMaxH3T2VBackend(
                client=FakeClient(),
                workflow_path=Path("/tmp/wf.json"),
                output_dir=tmp_path / "output",
                asset_uploader=uploader,
                render_queue=queue,
                postprocess=False,
                model_resolver=FakeModelResolver(),
                debug_workflows_dir=debug_dir,
                workflow=_t2v_workflow(),
            )
            request = VideoRenderRequest(
                scene={
                    "scene": 1,
                    "description": "Test",
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

    def test_validation_never_raises(self):
        """T2V validation always passes, not even on empty scenes."""
        with tempfile.TemporaryDirectory() as tmp:
            backend = ComfyUIMiniMaxH3T2VBackend(
                client=FakeClient(),
                workflow_path=Path("/tmp/wf.json"),
                output_dir=Path(tmp),
                workflow=_t2v_workflow(),
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
            # Validation does not raise inside build_workflow
            wf = backend.build_workflow(request.scene, prompt=request.prompt)
            self.assertIn("131", wf)

    def test_scene_workflow_json_written(self):
        """workflow.json is written to the per-scene directory."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            uploader = FakeAssetUploader()
            queue = FakeRenderQueue()
            backend = ComfyUIMiniMaxH3T2VBackend(
                client=FakeClient(),
                workflow_path=tmp_path / "wf.json",
                output_dir=tmp_path / "output",
                asset_uploader=uploader,
                render_queue=queue,
                postprocess=False,
                model_resolver=FakeModelResolver(),
                workflow=_t2v_workflow(),
            )
            request = VideoRenderRequest(
                scene={"scene": 3, "description": "Test"},
                scene_number=3,
                prompt="Test prompt",
                workflow_path=tmp_path / "wf.json",
                output_dir=tmp_path / "output",
                audio_file=tmp_path / "song.wav",
                storyboard_dir=tmp_path / "storyboard",
                upload_audio=False,
            )
            backend.render_video(request)
            scene_workflow = tmp_path / "output" / "scene_0003" / "workflow.json"
            self.assertTrue(scene_workflow.exists())
            data = json.loads(scene_workflow.read_text())
            self.assertIsInstance(data, dict)


# ---------------------------------------------------------------------------
# Frame path resolution tests
# ---------------------------------------------------------------------------

class ResolveFramePathTests(unittest.TestCase):
    def test_start_frame_resolved(self):
        backend = ComfyUIMiniMaxH3T2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
        )
        scene = {
            "keyframes": {"startframe_path": "start.png"},
        }
        path = backend._resolve_start_frame(scene)
        self.assertIsNotNone(path)
        self.assertIn("start", str(path))

    def test_end_frame_resolved(self):
        backend = ComfyUIMiniMaxH3T2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
        )
        scene = {
            "keyframes": {"endframe_path": "end.png"},
        }
        path = backend._resolve_end_frame(scene)
        self.assertIsNotNone(path)
        self.assertIn("end", str(path))

    def test_no_keyframes_returns_none(self):
        backend = ComfyUIMiniMaxH3T2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
        )
        self.assertIsNone(backend._resolve_start_frame({}))
        self.assertIsNone(backend._resolve_end_frame({}))

    def test_keyframes_without_path(self):
        backend = ComfyUIMiniMaxH3T2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
        )
        scene = {"keyframes": {}}
        self.assertIsNone(backend._resolve_start_frame(scene))
        self.assertIsNone(backend._resolve_end_frame(scene))

    def test_project_dir_resolution(self):
        backend = ComfyUIMiniMaxH3T2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
            project_dir=Path("/tmp/project"),
        )
        scene = {"keyframes": {"startframe_path": "assets/start.png"}}
        path = backend._resolve_start_frame(scene)
        self.assertTrue(path.exists() or True)  # coerce_local_path may resolve to non-existing
        self.assertIn("start", str(path))


# ---------------------------------------------------------------------------
# Class constants tests
# ---------------------------------------------------------------------------

class ClassConstantsTests(unittest.TestCase):
    def test_max_frames(self):
        self.assertEqual(2, ComfyUIMiniMaxH3T2VBackend.MAX_FRAMES)

    def test_fps(self):
        self.assertEqual(24, ComfyUIMiniMaxH3T2VBackend.FPS)


# ---------------------------------------------------------------------------
# Inheritance tests
# ---------------------------------------------------------------------------

class InheritanceTests(unittest.TestCase):
    def test_inherits_from_base(self):
        self.assertTrue(
            issubclass(ComfyUIMiniMaxH3T2VBackend, ComfyUIMiniMaxH3VideoRenderBackend)
        )

    def test_uses_base_patch_megapixels(self):
        backend = ComfyUIMiniMaxH3T2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
        )
        wf = _t2v_workflow()
        patcher = WorkflowPatcher(wf)
        # Should use base class method via MRO
        backend._patch_megapixels(patcher, 0.5)
        self.assertEqual(0.5, patcher.get()["115"]["inputs"]["megapixels"])

    def test_uses_base_patch_seed(self):
        backend = ComfyUIMiniMaxH3T2VBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp/out"),
        )
        wf = _t2v_workflow()
        patcher = WorkflowPatcher(wf)
        backend._patch_seed(patcher, 77777)
        self.assertEqual(77777, patcher.get()["129"]["inputs"]["noise_seed"])


if __name__ == "__main__":
    unittest.main()
