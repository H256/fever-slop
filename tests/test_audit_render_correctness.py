import json
import tempfile
import unittest
from pathlib import Path


class AuditRenderCorrectnessTests(unittest.TestCase):
    def test_missing_movie_visual_plan_has_contextual_error(self):
        from feverslop.application.movie_i2v_render_plan import write_movie_i2v_render_plan

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FileNotFoundError, "visual_plan.json"):
                write_movie_i2v_render_plan(project_dir=Path(temp_dir))

    def test_local_startframe_validation_is_not_marked_as_passed(self):
        from feverslop.application.startframe_validation import write_local_startframe_validation

        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            path = project / "movie" / "startframe_plan.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"shots": [{"scene": 1, "shot_id": "shot_001"}]}))

            output = write_local_startframe_validation(project_dir=project)
            payload = json.loads(output.read_text())
            self.assertFalse(payload["shots"][0]["pass"])


if __name__ == "__main__":
    unittest.main()
