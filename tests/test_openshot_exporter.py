import json
import xml.etree.ElementTree as ET
from unittest.mock import patch
import tempfile
import unittest
from pathlib import Path


def _write_overlapping_plan(root):
    plan = root / "plan.json"
    plan.write_text(json.dumps([
        {"scene": 1, "duration_seconds": 2.0, "abs_start_seconds": 0.0},
        {"scene": 2, "duration_seconds": 0.4, "abs_start_seconds": 0.5},
    ]), encoding="utf-8")
    clips = [root / "scene_0001.mp4", root / "scene_0002.mp4"]
    for clip in clips:
        clip.touch()
    return plan, clips


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

    def test_rejects_empty_render_plan(self):
        from feverslop.application.openshot_exporter import export_render_plan_to_openshot

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = root / "plan.json"
            plan.write_text(json.dumps([]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Render plan is empty:"):
                export_render_plan_to_openshot(
                    render_plan_path=plan,
                    clip_paths=[],
                    output_path=root / "movie.osp",
                    width=1216,
                    height=672,
                    fps=24,
                )

    def test_rejects_overlapping_absolute_render_plan_entries(self):
        from feverslop.application.mlt_exporter import export_render_plan_to_mlt
        from feverslop.application.openshot_exporter import export_render_plan_to_openshot

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan, clips = _write_overlapping_plan(root)

            with self.assertRaises(ValueError) as openshot_error:
                export_render_plan_to_openshot(
                    render_plan_path=plan,
                    clip_paths=clips,
                    output_path=root / "movie.osp",
                    width=1216,
                    height=672,
                    fps=24,
                )
            with self.assertRaises(ValueError) as mlt_error:
                export_render_plan_to_mlt(
                    render_plan_path=plan,
                    clip_paths=clips,
                    output_path=root / "timeline.mlt",
                    width=1216,
                    height=672,
                    fps=24,
                )

        self.assertEqual(str(openshot_error.exception), str(mlt_error.exception))
        self.assertIn("scene 2 ends at frame 22, before frame 48", str(openshot_error.exception))

    def test_tolerates_one_frame_boundary_rounding_difference(self):
        from feverslop.application.openshot_exporter import export_render_plan_to_openshot

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

            export_render_plan_to_openshot(
                render_plan_path=plan,
                clip_paths=clips,
                output_path=root / "movie.osp",
                width=1216,
                height=672,
                fps=24,
            )

            project = json.loads((root / "movie.osp").read_text(encoding="utf-8"))
            self.assertEqual([clip["position"] for clip in project["clips"]], [0.0, 23 / 24])
            self.assertEqual(project["duration"], 23 / 24 + 1.0)

    def test_repairs_accumulated_rounding_drift_with_absolute_plan(self):
        from feverslop.application.openshot_exporter import export_render_plan_to_openshot

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = root / "plan.json"
            plan.write_text(json.dumps([
                {"scene": 1, "duration_seconds": 1.0217, "abs_start_seconds": 0.0},
                {"scene": 2, "duration_seconds": 1.0433, "abs_start_seconds": 1.02},
                {"scene": 3, "duration_seconds": 1.0, "abs_start_seconds": 2.0616},
            ]), encoding="utf-8")
            clips = [root / "one.mp4", root / "two.mp4", root / "three.mp4"]
            for clip in clips:
                clip.touch()

            export_render_plan_to_openshot(
                render_plan_path=plan,
                clip_paths=clips,
                output_path=root / "movie.osp",
                width=1216,
                height=672,
                fps=24,
            )

            self.assertTrue((root / "movie.osp").is_file())

    def test_mlt_orders_absolute_entries_by_timeline_position(self):
        from feverslop.application.mlt_exporter import export_render_plan_to_mlt

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = root / "plan.json"
            plan.write_text(json.dumps([
                {"scene": 2, "duration_seconds": 1.0, "abs_start_seconds": 1.0},
                {"scene": 1, "duration_seconds": 1.0, "abs_start_seconds": 0.0},
            ]), encoding="utf-8")
            clips = [root / "two.mp4", root / "one.mp4"]
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

            playlist = ET.parse(root / "timeline.mlt").getroot().find("playlist[@id='playlist0']")
            self.assertEqual(
                [entry.attrib["producer"] for entry in playlist.findall("entry")],
                ["video_0002", "video_0001"],
            )


class MltExporterTests(unittest.TestCase):
    def test_exports_ordered_video_and_original_audio_as_mlt_xml(self):
        from feverslop.application.mlt_exporter import export_render_plan_to_mlt

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = root / "render_plan.json"
            plan.write_text(json.dumps([
                {
                    "scene": 1,
                    "seed": 123,
                    "duration_seconds": 1.5,
                    "abs_start_seconds": 0.0,
                    "metadata": {
                        "base_concept": "A dwarf enters the cavern.",
                        "camera_motion": "Slow dolly in.",
                        "character_motion": "Raises the hammer.",
                    },
                },
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
                project_name="The Well of Youth",
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
            notes = root_element.find("property[@name='shotcut:projectNotes']").text
            self.assertIn("Project: The Well of Youth", notes)
            self.assertIn("Scenes: 2", notes)
            self.assertIn("Profile: 1216x672 @ 24 fps", notes)
            self.assertIn("Duration: 00:03", notes)
            self.assertIn("Audio track: A1 - Original audio (dwarfventure.mp3)", notes)
            self.assertEqual(
                root_element.find("chain[@id='video_0001']/property[@name='shotcut:caption']").text,
                "Scene 0001",
            )
            comment = root_element.find("chain[@id='video_0001']/property[@name='shotcut:comment']").text
            self.assertIn("Story: A dwarf enters the cavern.", comment)
            self.assertIn("Camera: Slow dolly in.", comment)
            self.assertIn("Character: Raises the hammer.", comment)
            self.assertIn("Seed: 123", comment)
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
            self.assertEqual([], [p for p in root.rglob("*") if p.name.endswith(".tmp")])

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

    def test_repairs_accumulated_rounding_drift_with_absolute_plan(self):
        from feverslop.application.mlt_exporter import export_render_plan_to_mlt

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = root / "plan.json"
            plan.write_text(json.dumps([
                {"scene": 1, "duration_seconds": 1.0217, "abs_start_seconds": 0.0},
                {"scene": 2, "duration_seconds": 1.0433, "abs_start_seconds": 1.02},
                {"scene": 3, "duration_seconds": 1.0, "abs_start_seconds": 2.0616},
            ]), encoding="utf-8")
            clips = [root / "one.mp4", root / "two.mp4", root / "three.mp4"]
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
            video_playlist = root_element.find("playlist[@id='playlist0']")
            entries = video_playlist.findall("entry")
            self.assertEqual([entry.attrib["in"] for entry in entries], ["0", "0", "0"])
            self.assertEqual([entry.attrib["out"] for entry in entries], ["24", "24", "22"])
            self.assertEqual(root_element.find("tractor[@id='main']").attrib["out"], "72")

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

    def test_rejects_empty_render_plan(self):
        from feverslop.application.mlt_exporter import export_render_plan_to_mlt

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = root / "plan.json"
            plan.write_text(json.dumps([]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Render plan is empty:"):
                export_render_plan_to_mlt(
                    render_plan_path=plan,
                    clip_paths=[],
                    output_path=root / "timeline.mlt",
                    width=1216,
                    height=672,
                    fps=24,
                )

    def test_rejects_empty_dict_render_plan(self):
        from feverslop.application.mlt_exporter import export_render_plan_to_mlt

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = root / "plan.json"
            plan.write_text(json.dumps({"shots": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Render plan is empty:"):
                export_render_plan_to_mlt(
                    render_plan_path=plan,
                    clip_paths=[],
                    output_path=root / "timeline.mlt",
                    width=1216,
                    height=672,
                    fps=24,
                )

    def test_rejects_overlapping_absolute_render_plan_entries(self):
        from feverslop.application.mlt_exporter import export_render_plan_to_mlt

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan, clips = _write_overlapping_plan(root)
            with self.assertRaises(ValueError) as error:
                export_render_plan_to_mlt(
                    render_plan_path=plan,
                    clip_paths=clips,
                    output_path=root / "timeline.mlt",
                    width=1216,
                    height=672,
                    fps=24,
                )
            message = str(error.exception)
            self.assertIn(
                "Timeline export cannot represent overlapping render-plan entries: ",
                message,
            )
            self.assertIn("scene 2 ends at frame 22, before frame 48", message)
            self.assertIn(f"Render plan: {plan}", message)


if __name__ == "__main__":
    unittest.main()
