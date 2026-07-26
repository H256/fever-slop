import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from feverslop.application.facefix_pipeline import FaceFixPipelineStep, FaceFixRequest
from feverslop.domain.facefix_rendering import FaceFixConfig


class TestFaceFixPipelineStep(unittest.TestCase):
    def setUp(self):
        self.mock_backend = MagicMock()
        self.mock_backend.render_scene.return_value = Path("/tmp/scene_0001_facefix.mp4")
        self.mock_reporter = MagicMock()

    def test_execute_runs_facefix_for_all_scenes(self):
        step = FaceFixPipelineStep(
            backend=self.mock_backend,
            reporter=self.mock_reporter,
        )

        exists_calls = 0
        original_exists = Path.exists

        def mock_exists(self):
            nonlocal exists_calls
            exists_calls += 1
            if "rendered" in str(self) or "output" in str(self):
                return True
            return original_exists(self)

        with patch.object(Path, "exists", mock_exists):
            results = step.execute(FaceFixRequest(
                rendered_dir=Path("/tmp/rendered"),
                output_dir=Path("/tmp/output"),
                scene_numbers=[1, 2],
                reference_images=[Path("/tmp/face.png")],
                skip_existing=False,
            ))

            self.assertEqual(len(results), 2)
            self.assertEqual(self.mock_backend.render_scene.call_count, 2)

    def test_execute_skips_missing_sources(self):
        step = FaceFixPipelineStep(
            backend=self.mock_backend,
            reporter=self.mock_reporter,
        )

        def mock_exists(self):
            if "face.png" in str(self):
                return True
            if "output" in str(self):
                return True
            return False

        with patch.object(Path, "exists", mock_exists):
            results = step.execute(FaceFixRequest(
                rendered_dir=Path("/tmp/rendered"),
                output_dir=Path("/tmp/output"),
                scene_numbers=[1, 2],
                reference_images=[Path("/tmp/face.png")],
            ))
            self.assertEqual(len(results), 0)
            self.assertEqual(self.mock_backend.render_scene.call_count, 0)

    def test_execute_skips_existing(self):
        step = FaceFixPipelineStep(
            backend=self.mock_backend,
            reporter=self.mock_reporter,
        )

        with patch.object(Path, "exists", return_value=True):
            results = step.execute(FaceFixRequest(
                rendered_dir=Path("/tmp/rendered"),
                output_dir=Path("/tmp/output"),
                scene_numbers=[1],
                skip_existing=True,
            ))
            self.assertEqual(len(results), 1)
            self.assertEqual(self.mock_backend.render_scene.call_count, 0)

    def test_default_reporter_is_null(self):
        step = FaceFixPipelineStep(backend=self.mock_backend)
        from feverslop.ports.reporting import NullReporter
        self.assertIsInstance(step.reporter, NullReporter)


if __name__ == "__main__":
    unittest.main()
