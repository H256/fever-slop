import unittest
import tempfile
from pathlib import Path

from feverslop.adapters.comfyui_minimax_h3_video_backend import (
    ComfyUIMiniMaxH3VideoRenderBackend,
)
from feverslop.adapters.workflow_patcher import WorkflowPatcher
from feverslop.domain.postprocessing import TrimSpec
from feverslop.errors import FeverSlopValidationError


class FakeClient:
    pass


class FakeAssetUploader:
    def __init__(self):
        self.calls = []


class FakeRenderQueue:
    def __init__(self):
        self.calls = []


class FakePostProcessor:
    def __init__(self):
        self.trim_specs = []

    def trim_clip(self, spec):
        self.trim_specs.append(spec)
        spec.output_file.parent.mkdir(parents=True, exist_ok=True)
        spec.output_file.write_bytes(b"final")
        return spec.output_file


class FakeModelResolver:
    def resolve_workflow_models(self, workflow, workflow_path=None):
        return workflow


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------

class ConstructorTests(unittest.TestCase):
    def test_constructor_stores_params(self):
        backend = ComfyUIMiniMaxH3VideoRenderBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/workflow.json"),
            output_dir=Path("/tmp/output"),
        )
        self.assertIsInstance(backend.client, FakeClient)
        self.assertEqual(Path("/tmp/output"), backend.output_dir)
        self.assertEqual(Path("/tmp/output") / "raw", backend.raw_output_dir)

    def test_constructor_creates_default_dependencies(self):
        backend = ComfyUIMiniMaxH3VideoRenderBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/workflow.json"),
            output_dir=Path("/tmp/output"),
        )
        self.assertIsNotNone(backend.asset_uploader)
        self.assertIsNotNone(backend.render_queue)
        self.assertIsNotNone(backend.postprocessor)
        self.assertIsNone(backend.model_resolver)

    def test_constructor_accepts_injected_dependencies(self):
        uploader = FakeAssetUploader()
        queue = FakeRenderQueue()
        postprocessor = FakePostProcessor()
        resolver = FakeModelResolver()
        backend = ComfyUIMiniMaxH3VideoRenderBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/workflow.json"),
            output_dir=Path("/tmp/output"),
            asset_uploader=uploader,
            render_queue=queue,
            postprocessor=postprocessor,
            model_resolver=resolver,
        )
        self.assertIs(backend.asset_uploader, uploader)
        self.assertIs(backend.render_queue, queue)
        self.assertIs(backend.postprocessor, postprocessor)
        self.assertIs(backend.model_resolver, resolver)

    def test_constructor_clamps_frames(self):
        backend = ComfyUIMiniMaxH3VideoRenderBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/workflow.json"),
            output_dir=Path("/tmp/output"),
            preroll_frames=-5,
            tail_loss_frames=-10,
        )
        self.assertEqual(0, backend.preroll_frames)
        self.assertEqual(0, backend.tail_loss_frames)

    def test_constructor_stores_in_memory_workflow(self):
        wf = {"1": {"class_type": "MiniMaxH3Video", "inputs": {}}}
        backend = ComfyUIMiniMaxH3VideoRenderBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/workflow.json"),
            output_dir=Path("/tmp/output"),
            workflow=wf,
        )
        self.assertIsNotNone(backend.workflow)
        self.assertEqual({"1": {"class_type": "MiniMaxH3Video", "inputs": {}}}, backend.workflow)


# ---------------------------------------------------------------------------
# load_workflow
# ---------------------------------------------------------------------------

class LoadWorkflowTests(unittest.TestCase):
    def test_load_workflow_from_memory(self):
        wf = {"1": {"class_type": "MiniMaxH3Video", "inputs": {"prompt": ""}}}
        backend = ComfyUIMiniMaxH3VideoRenderBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/workflow.json"),
            output_dir=Path("/tmp/output"),
            workflow=wf,
        )
        loaded = backend.load_workflow()
        self.assertEqual(wf, loaded)
        # Should be a deep copy, not the same object
        self.assertIsNot(loaded, wf)

    def test_load_workflow_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            wf_path = Path(tmp) / "workflow.json"
            wf_path.write_text('{"1": {"class_type": "Test"}}', encoding="utf-8")
            backend = ComfyUIMiniMaxH3VideoRenderBackend(
                client=FakeClient(),
                workflow_path=wf_path,
                output_dir=Path(tmp),
            )
            loaded = backend.load_workflow()
            self.assertEqual({"1": {"class_type": "Test"}}, loaded)


# ---------------------------------------------------------------------------
# _frames_from_duration
# ---------------------------------------------------------------------------

class FramesFromDurationTests(unittest.TestCase):
    def test_4_seconds(self):
        self.assertEqual(103, ComfyUIMiniMaxH3VideoRenderBackend._frames_from_duration(4.0))

    def test_5_seconds(self):
        self.assertEqual(120, ComfyUIMiniMaxH3VideoRenderBackend._frames_from_duration(5.0))

    def test_10_seconds(self):
        self.assertEqual(256, ComfyUIMiniMaxH3VideoRenderBackend._frames_from_duration(10.0))

    def test_15_seconds(self):
        self.assertEqual(375, ComfyUIMiniMaxH3VideoRenderBackend._frames_from_duration(15.0))


# ---------------------------------------------------------------------------
# _validate_resolution
# ---------------------------------------------------------------------------

class ValidateResolutionTests(unittest.TestCase):
    def test_1344x768_ok(self):
        ComfyUIMiniMaxH3VideoRenderBackend._validate_resolution(1344, 768)

    def test_2048x2048_ok(self):
        ComfyUIMiniMaxH3VideoRenderBackend._validate_resolution(2048, 2048)

    def test_512x512_ok(self):
        ComfyUIMiniMaxH3VideoRenderBackend._validate_resolution(512, 512)

    def test_over_max_width_raises(self):
        with self.assertRaises(FeverSlopValidationError):
            ComfyUIMiniMaxH3VideoRenderBackend._validate_resolution(2049, 1024)

    def test_over_max_height_raises(self):
        with self.assertRaises(FeverSlopValidationError):
            ComfyUIMiniMaxH3VideoRenderBackend._validate_resolution(1024, 2049)

    def test_under_min_width_raises(self):
        with self.assertRaises(FeverSlopValidationError):
            ComfyUIMiniMaxH3VideoRenderBackend._validate_resolution(511, 1024)

    def test_under_min_height_raises(self):
        with self.assertRaises(FeverSlopValidationError):
            ComfyUIMiniMaxH3VideoRenderBackend._validate_resolution(1024, 511)


# ---------------------------------------------------------------------------
# _patch_minimax_core
# ---------------------------------------------------------------------------

class PatchMiniMaxCoreTests(unittest.TestCase):
    def test_patches_minimax_video_node(self):
        workflow = {
            "1": {
                "class_type": "MiniMaxH3Video",
                "inputs": {"prompt": "", "width": 0, "height": 0, "length": 0},
            },
        }
        patcher = WorkflowPatcher(workflow)
        ComfyUIMiniMaxH3VideoRenderBackend._patch_minimax_core(
            patcher, "test prompt", 1344, 768, 103
        )
        node = patcher.get()["1"]
        self.assertEqual("test prompt", node["inputs"]["prompt"])
        self.assertEqual(1344, node["inputs"]["width"])
        self.assertEqual(768, node["inputs"]["height"])
        self.assertEqual(103, node["inputs"]["length"])

    def test_patches_r2v_node(self):
        workflow = {
            "1": {
                "class_type": "MiniMaxH3ReferenceToVideo",
                "inputs": {"prompt": "", "width": 0, "height": 0, "length": 0},
            },
        }
        patcher = WorkflowPatcher(workflow)
        ComfyUIMiniMaxH3VideoRenderBackend._patch_minimax_core(
            patcher, "r2v prompt", 1024, 1024, 120
        )
        node = patcher.get()["1"]
        self.assertEqual("r2v prompt", node["inputs"]["prompt"])
        self.assertEqual(1024, node["inputs"]["width"])
        self.assertEqual(1024, node["inputs"]["height"])
        self.assertEqual(120, node["inputs"]["length"])

    def test_prefers_t2v_over_r2v(self):
        """MiniMaxH3Video takes priority when both are present."""
        workflow = {
            "1": {
                "class_type": "MiniMaxH3Video",
                "inputs": {"prompt": "", "width": 0, "height": 0, "length": 0},
            },
            "2": {
                "class_type": "MiniMaxH3ReferenceToVideo",
                "inputs": {"prompt": "", "width": 0, "height": 0, "length": 0},
            },
        }
        patcher = WorkflowPatcher(workflow)
        ComfyUIMiniMaxH3VideoRenderBackend._patch_minimax_core(
            patcher, "the prompt", 1024, 1024, 103
        )
        # Node 2 should remain unpatched (zeros)
        node2 = patcher.get()["2"]
        self.assertEqual("", node2["inputs"]["prompt"])
        self.assertEqual(0, node2["inputs"]["width"])

    def test_raises_when_neither_node_present(self):
        workflow = {
            "1": {"class_type": "SomeOtherNode", "inputs": {}},
        }
        patcher = WorkflowPatcher(workflow)
        with self.assertRaises(KeyError) as ctx:
            ComfyUIMiniMaxH3VideoRenderBackend._patch_minimax_core(
                patcher, "prompt", 512, 512, 1
            )
        self.assertIn("MiniMaxH3Video", str(ctx.exception))
        self.assertIn("MiniMaxH3ReferenceToVideo", str(ctx.exception))


# ---------------------------------------------------------------------------
# _decode_packed_latent
# ---------------------------------------------------------------------------

class DecodePackedLatentTests(unittest.TestCase):
    def test_passes_when_both_nodes_exist(self):
        workflow = {
            "1": {"class_type": "VAEDecode", "inputs": {}},
            "2": {"class_type": "VAEDecodeAudio", "inputs": {}},
        }
        patcher = WorkflowPatcher(workflow)
        ComfyUIMiniMaxH3VideoRenderBackend._decode_packed_latent(patcher)

    def test_fails_when_missing_vae_decode(self):
        workflow = {
            "1": {"class_type": "VAEDecodeAudio", "inputs": {}},
        }
        patcher = WorkflowPatcher(workflow)
        with self.assertRaises(KeyError) as ctx:
            ComfyUIMiniMaxH3VideoRenderBackend._decode_packed_latent(patcher)
        self.assertIn("VAEDecode", str(ctx.exception))

    def test_fails_when_missing_vae_decode_audio(self):
        workflow = {
            "1": {"class_type": "VAEDecode", "inputs": {}},
        }
        patcher = WorkflowPatcher(workflow)
        with self.assertRaises(KeyError) as ctx:
            ComfyUIMiniMaxH3VideoRenderBackend._decode_packed_latent(patcher)
        self.assertIn("VAEDecodeAudio", str(ctx.exception))


# ---------------------------------------------------------------------------
# _postprocess_with_audio
# ---------------------------------------------------------------------------

class PostprocessWithAudioTests(unittest.TestCase):
    def test_delegates_to_postprocessor(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw = tmp_path / "raw.mp4"
            raw.write_bytes(b"raw video")
            postprocessor = FakePostProcessor()
            backend = ComfyUIMiniMaxH3VideoRenderBackend(
                client=FakeClient(),
                workflow_path=Path("/tmp/wf.json"),
                output_dir=tmp_path,
                postprocessor=postprocessor,
            )
            spec = TrimSpec(
                source_file=raw,
                output_file=tmp_path / "final.mp4",
                fps=24,
                trim_front_frames=0,
                keep_frames=100,
                scene=1,
            )
            result = backend._postprocess_with_audio(raw, spec)
            self.assertEqual(tmp_path / "final.mp4", result)
            self.assertEqual(1, len(postprocessor.trim_specs))
            self.assertIs(postprocessor.trim_specs[0], spec)


# ---------------------------------------------------------------------------
# Abstract methods
# ---------------------------------------------------------------------------

class AbstractMethodsTests(unittest.TestCase):
    def test_build_workflow_raises_not_implemented(self):
        backend = ComfyUIMiniMaxH3VideoRenderBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp"),
        )
        with self.assertRaises(NotImplementedError):
            backend.build_workflow({})

    def test_render_video_raises_not_implemented(self):
        backend = ComfyUIMiniMaxH3VideoRenderBackend(
            client=FakeClient(),
            workflow_path=Path("/tmp/wf.json"),
            output_dir=Path("/tmp"),
        )
        with self.assertRaises(NotImplementedError):
            backend.render_video({})


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class ConstantsTests(unittest.TestCase):
    def test_min_dimension(self):
        self.assertEqual(512, ComfyUIMiniMaxH3VideoRenderBackend.MIN_DIMENSION)

    def test_max_dimension(self):
        self.assertEqual(2048, ComfyUIMiniMaxH3VideoRenderBackend.MAX_DIMENSION)


if __name__ == "__main__":
    unittest.main()
