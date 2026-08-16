import json
from unittest.mock import patch
import tempfile
import unittest
from pathlib import Path


class OpenShotExporterTests(unittest.TestCase):
    @patch("subprocess.run")
    def test_uses_rendered_clip_profile_for_project(self, run_probe):
        from feverslop.application.openshot_exporter import export_render_plan_to_openshot

        run_probe.return_value = type("Completed", (), {
            "stdout": json.dumps({"streams": [{"width": 1216, "height": 672, "r_frame_rate": "24/1"}]}),
        })()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = root / "plan.json"
            plan.write_text(json.dumps([{"scene": 1, "duration_seconds": 2.0}]), encoding="utf-8")
            clip = root / "scene_0001.mp4"
            clip.touch()
            output = root / "movie.osp"

            export_render_plan_to_openshot(
                render_plan_path=plan,
                clip_paths=[clip],
                output_path=output,
                width=1280,
                height=720,
                fps=30,
            )

            project = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual((project["width"], project["height"]), (1216, 672))
            self.assertEqual(project["fps"], {"num": 24, "den": 1})
            self.assertEqual(project["files"][0]["width"], 1216)
            self.assertEqual(project["files"][0]["height"], 672)
            self.assertEqual(project["files"][0]["fps"], {"num": 24, "den": 1})

    def test_exports_relative_video_and_audio_clips_in_render_plan_order(self):
        from feverslop.application.openshot_exporter import export_render_plan_to_openshot

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan_path = root / "render_plan.json"
            plan_path.write_text(
                json.dumps([
                    {"scene": 2, "duration_seconds": 3.5, "abs_start_seconds": 3.5},
                    {"scene": 1, "duration_seconds": 3.5, "abs_start_seconds": 0.0},
                ]),
                encoding="utf-8",
            )
            first = root / "render" / "scene_0002.mp4"
            second = root / "render" / "scene_0001.mp4"
            audio = root / "audio" / "song.wav"
            for path in (first, second, audio):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            output = root / "output" / "openshot" / "movie.osp"

            result = export_render_plan_to_openshot(
                render_plan_path=plan_path,
                clip_paths=[first, second],
                audio_path=audio,
                output_path=output,
                width=672,
                height=1216,
                fps=24,
            )

            self.assertEqual(result, output)
            project = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual((project["width"], project["height"]), (672, 1216))
            self.assertEqual(project["duration"], 7.0)
            self.assertEqual([clip["file_id"] for clip in project["clips"][:2]], ["file_video_0002", "file_video_0001"])
            self.assertEqual([clip["position"] for clip in project["clips"][:2]], [3.5, 0.0])
            self.assertEqual(project["clips"][2]["layer"], 2000000)
            self.assertEqual(project["clips"][0]["gravity"], 4)
            self.assertEqual(project["clips"][0]["scale"], 1)
            self.assertEqual(project["clips"][0]["reader"]["type"], "FFmpegReader")
            self.assertEqual(project["clips"][0]["alpha"]["Points"][0]["co"], {"X": 1.0, "Y": 1.0})
            self.assertEqual(project["clips"][0]["volume"]["Points"][0]["co"], {"X": 1.0, "Y": 1.0})
            self.assertEqual(project["version"], {"openshot-qt": "3.5.1", "libopenshot": "0.7.0"})
            self.assertTrue(all(not Path(file["path"]).is_absolute() for file in project["files"]))

    def test_reports_progress_for_each_video_clip_and_audio(self):
        from feverslop.application.openshot_exporter import export_render_plan_to_openshot

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = root / "plan.json"
            plan.write_text(json.dumps([{"scene": 1, "duration_seconds": 2.0}]), encoding="utf-8")
            clip = root / "scene_0001.mp4"
            audio = root / "song.wav"
            clip.touch()
            audio.touch()
            progress = []

            export_render_plan_to_openshot(
                render_plan_path=plan,
                clip_paths=[clip],
                audio_path=audio,
                output_path=root / "movie.osp",
                width=1280,
                height=720,
                fps=24,
                on_progress=lambda completed, total, label: progress.append((completed, total, label)),
            )

            self.assertEqual(progress, [(1, 2, "scene 1"), (2, 2, "audio")])


if __name__ == "__main__":
    unittest.main()
