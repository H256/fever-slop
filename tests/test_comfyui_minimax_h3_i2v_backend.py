import unittest
from pathlib import Path

from feverslop.adapters.comfyui_minimax_h3_i2v_backend import ComfyUIMiniMaxH3I2VBackend
from feverslop.errors import FeverSlopValidationError
from tests.test_comfyui_minimax_h3_t2v_backend import FakeClient


def _make_backend(tmp: Path) -> ComfyUIMiniMaxH3I2VBackend:
    return ComfyUIMiniMaxH3I2VBackend(
        client=FakeClient(),
        workflow_path=tmp / "workflow.json",
        output_dir=tmp / "output",
    )


class I2VBackendFrameValidationTests(unittest.TestCase):
    def test_missing_start_frame_raises_validation_error(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            backend = _make_backend(Path(tmp))
            with self.assertRaises(FeverSlopValidationError) as ctx:
                backend._validate_i2v_frames({"scene": 1, "keyframes": {}})
            self.assertIn("start frame", str(ctx.exception))

    def test_start_only_scene_passes(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            backend = _make_backend(Path(tmp))
            backend._validate_i2v_frames(
                {"scene": 1, "keyframes": {"startframe_path": "a.png"}}
            )

    def test_start_end_scene_passes(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            backend = _make_backend(Path(tmp))
            backend._validate_i2v_frames(
                {
                    "scene": 1,
                    "keyframes": {"startframe_path": "a.png", "endframe_path": "b.png"},
                }
            )

    def test_build_workflow_validates_frames(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            backend = _make_backend(Path(tmp))
            with self.assertRaises(FeverSlopValidationError):
                backend.build_workflow({"scene": 1, "keyframes": {}}, prompt="x")


if __name__ == "__main__":
    unittest.main()
