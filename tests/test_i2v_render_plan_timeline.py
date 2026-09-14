import json
import tempfile
import unittest
from pathlib import Path

from feverslop.application.movie_i2v_render_plan import write_movie_i2v_render_plan
from feverslop.application.render_plan_validation import validate_render_plan_timeline
from feverslop.application.startframe_i2v_render_plan import (
    write_startframe_i2v_render_plan,
)

# A duration whose rounded frame count (round(d * fps)) is not an integer
# multiple of d, so float accumulation of d per scene drifts away from the
# rendered frame cursor over many scenes.
FPS = 24
SCENE_DURATION = 4.3
SCENE_COUNT = 220


class I2VRenderPlanTimelineTests(unittest.TestCase):
    def _assert_frame_exact_plan(self, plan_path: Path) -> None:
        entries = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertEqual(SCENE_COUNT, len(entries))
        expected_frame_count = max(1, round(SCENE_DURATION * FPS))
        for index, scene in enumerate(entries):
            start_frame = max(0, round(float(scene["abs_start_seconds"]) * FPS))
            self.assertEqual(
                index * expected_frame_count,
                start_frame,
                f"scene {index + 1} starts at frame {start_frame}, "
                f"expected the frame-exact cursor {index * expected_frame_count}",
            )
            self.assertEqual(expected_frame_count, scene["frame_count"])
        # The acceptance criterion: the validation run needs no repair for
        # 200+ scenes, and a contiguous timeline validates without errors.
        validate_render_plan_timeline(entries, fps=FPS, render_plan_path=plan_path)

    def test_movie_i2v_plan_keeps_frame_exact_starts_over_200_scenes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            movie = project / "movie"
            movie.mkdir(parents=True)
            (movie / "visual_plan.json").write_text(
                json.dumps(
                    {
                        "shots": [
                            {
                                "scene": index,
                                "shot_id": f"shot_{index:04d}",
                                "view_id": "v1",
                                "duration_seconds": SCENE_DURATION,
                                "video_prompt": f"prompt {index}",
                                "base_plate_prompt": f"base {index}",
                            }
                            for index in range(1, SCENE_COUNT + 1)
                        ]
                    }
                ),
                encoding="utf-8",
            )

            plan_path = write_movie_i2v_render_plan(project_dir=project, fps=FPS)
            self._assert_frame_exact_plan(plan_path)

    def test_startframe_i2v_plan_keeps_frame_exact_starts_over_200_scenes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            movie = project / "movie"
            movie.mkdir(parents=True)
            (movie / "startframe_plan.json").write_text(
                json.dumps(
                    {
                        "shots": [
                            {
                                "scene": index,
                                "shot_id": f"shot_{index:04d}",
                                "duration_seconds": SCENE_DURATION,
                                "width": 1280,
                                "height": 704,
                                "ltx_motion": {"prompt": f"motion {index}"},
                                "startframe_intent": {"action_moment": f"moment {index}"},
                            }
                            for index in range(1, SCENE_COUNT + 1)
                        ]
                    }
                ),
                encoding="utf-8",
            )

            plan_path = write_startframe_i2v_render_plan(project_dir=project, fps=FPS)
            self._assert_frame_exact_plan(plan_path)


if __name__ == "__main__":
    unittest.main()
