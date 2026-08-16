import json
import xml.etree.ElementTree as ET
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


class MltExporterTests(unittest.TestCase):
    def test_exports_ordered_video_and_original_audio_as_mlt_xml(self):
        from feverslop.application.mlt_exporter import export_render_plan_to_mlt

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = root / "render_plan.json"
            plan.write_text(json.dumps([
                {"scene": 1, "duration_seconds": 1.5, "abs_start_seconds": 0.0},
                {"scene": 2, "duration_seconds": 1.5, "abs_start_seconds": 1.5},
            ]), encoding="utf-8")
            clips = [root / "scene_0001.mp4", root / "scene_0002.mp4"]
            audio = root / "dwarfventure.mp3"
            for path in (*clips, audio):
                path.touch()
            output = root / "shotcut" / "timeline.mlt"

            result = export_render_plan_to_mlt(
                render_plan_path=plan,
                clip_paths=clips,
                audio_path=audio,
                output_path=output,
                width=1216,
                height=672,
                fps=24,
            )

            self.assertEqual(result, output)
            document = ET.parse(output)
            root_element = document.getroot()
            self.assertEqual(root_element.attrib["producer"], "main_bin")
            self.assertIsNotNone(root_element.find("playlist[@id='main_bin']"))
            self.assertIsNone(root_element.find("playlist[@id='main bin']"))
            profile = root_element.find("profile")
            self.assertEqual(profile.attrib["width"], "1216")
            self.assertEqual(profile.attrib["height"], "672")
            self.assertEqual(profile.attrib["frame_rate_num"], "24")
            video_playlist = root_element.find("playlist[@id='playlist0']")
            audio_playlist = root_element.find("playlist[@id='playlist1']")
            children = list(root_element)
            self.assertLess(
                children.index(root_element.find("chain[@id='video_0001']")),
                children.index(video_playlist),
            )
            self.assertLess(
                children.index(root_element.find("chain[@id='audio_original']")),
                children.index(audio_playlist),
            )
            self.assertEqual(
                [entry.attrib["producer"] for entry in video_playlist.findall("entry")],
                ["video_0001", "video_0002"],
            )
            self.assertEqual(audio_playlist.find("entry").attrib["producer"], "audio_original")
            self.assertIsNotNone(root_element.find("playlist[@id='background']"))
            self.assertEqual(
                root_element.find("tractor/track").attrib["producer"],
                "background",
            )
            self.assertIsNone(root_element.find("tractor/multitrack"))
            self.assertEqual(
                [track.attrib["producer"] for track in root_element.findall("tractor/track")],
                ["background", "playlist0", "playlist1"],
            )
            self.assertEqual(
                root_element.find("tractor/track[@producer='playlist1']").attrib["hide"],
                "video",
            )

    def test_preserves_gaps_from_absolute_render_plan_positions(self):
        from feverslop.application.mlt_exporter import export_render_plan_to_mlt

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = root / "plan.json"
            plan.write_text(json.dumps([
                {"scene": 1, "duration_seconds": 1.0, "abs_start_seconds": 0.0},
                {"scene": 2, "duration_seconds": 1.0, "abs_start_seconds": 2.0},
            ]), encoding="utf-8")
            clips = [root / "one.mp4", root / "two.mp4"]
            for clip in clips:
                clip.touch()
            export_render_plan_to_mlt(
                render_plan_path=plan,
                clip_paths=clips,
                output_path=root / "timeline.mlt",
                width=1216,
                height=672,
                fps=24,
            )
            root_element = ET.parse(root / "timeline.mlt").getroot()
            self.assertEqual(root_element.find("playlist[@id='playlist0']/blank").attrib["length"], "24")

    def test_tolerates_one_frame_boundary_rounding_difference(self):
        from feverslop.application.mlt_exporter import export_render_plan_to_mlt

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = root / "plan.json"
            plan.write_text(json.dumps([
                {"scene": 1, "duration_seconds": 1.0, "abs_start_seconds": 0.0},
                {"scene": 2, "duration_seconds": 1.0, "abs_start_seconds": 23 / 24},
            ]), encoding="utf-8")
            clips = [root / "one.mp4", root / "two.mp4"]
            for clip in clips:
                clip.touch()
            export_render_plan_to_mlt(
                render_plan_path=plan,
                clip_paths=clips,
                output_path=root / "timeline.mlt",
                width=1216,
                height=672,
                fps=24,
            )

    def test_rejects_mismatched_plan_and_clip_counts(self):
        from feverslop.application.mlt_exporter import export_render_plan_to_mlt

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = root / "plan.json"
            plan.write_text(json.dumps([{"scene": 1, "duration_seconds": 1.0}]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "one rendered clip"):
                export_render_plan_to_mlt(
                    render_plan_path=plan,
                    clip_paths=[],
                    output_path=root / "timeline.mlt",
                    width=1216,
                    height=672,
                    fps=24,
                )


if __name__ == "__main__":
    unittest.main()
