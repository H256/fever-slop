import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from feverslop.application.facefix_pipeline import FaceFixPipelineStep, FaceFixRequest


class TestFaceFixPipelineStep(unittest.TestCase):
    def setUp(self):
        self.mock_backend = MagicMock()
        self.mock_backend.render_scene.return_value = Path("/tmp/scene_0001/final_facefix.mp4")
        self.mock_reporter = MagicMock()

    def test_execute_runs_facefix_for_all_scenes(self):
        step = FaceFixPipelineStep(
            backend=self.mock_backend,
            reporter=self.mock_reporter,
        )

        def mock_exists(p):
            s = str(p).replace("\\", "/")
            if s.endswith("/final.mp4"):
                return True
            return False

        def mock_iterdir(p):
            class FakeDir:
                name = "scene_0001"
                def is_dir(self):
                    return True
            return [FakeDir()]

        with patch.object(Path, "exists", mock_exists):
            with patch.object(Path, "iterdir", mock_iterdir):
                results = step.execute(FaceFixRequest(
                    scenes_dir=Path("/tmp/scenes"),
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

        def mock_exists(p):
            return False

        with patch.object(Path, "exists", mock_exists):
            results = step.execute(FaceFixRequest(
                scenes_dir=Path("/tmp/scenes"),
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

        def mock_exists(p):
            return True

        with patch.object(Path, "exists", mock_exists):
            results = step.execute(FaceFixRequest(
                scenes_dir=Path("/tmp/scenes"),
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
