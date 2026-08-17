import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from feverslop.composition.seedvr2_pipeline import SeedVR2CompositionOptions, run_seedvr2
from feverslop.domain.seedvr2 import plan_seedvr2_segments
from feverslop.scene_artifacts import SceneArtifactLayout


class FakeBackend:
    def __init__(self):
        self.calls = []

    def render(self, **kwargs):
        self.calls.append(kwargs)
        output = Path(kwargs["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"video")
        return output


class RecordingReporter:
    def __init__(self):
        self.messages = []

    def message(self, text):
        self.messages.append(text)


class FakePostProcessor:
    def __init__(self):
        self.concat_calls = []

    def write_concat_list(self, video_files, output_file):
        Path(output_file).write_text("\n".join(str(path) for path in video_files), encoding="utf-8")
        return Path(output_file)

    def concat_clips(self, concat_list, output_file, video_only=False, reencode=False, fps=None, frame_count=None):
        self.concat_calls.append((Path(concat_list), Path(output_file), video_only, reencode, fps, frame_count))
        output = Path(output_file)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"concatenated video")
        return output


class SeedVR2PipelineTests(unittest.TestCase):
    def test_segment_planner_balances_tiny_remainder_into_previous_segment(self):
        segments = plan_seedvr2_segments(5.29, max_segment_duration=5.0)

        self.assertEqual(2, len(segments))
        self.assertAlmostEqual(2.645, segments[0].duration_seconds, places=3)
        self.assertAlmostEqual(2.645, segments[1].duration_seconds, places=3)
        self.assertAlmostEqual(5.29, sum(segment.duration_seconds for segment in segments), places=3)

    def test_run_seedvr2_creates_multi_pass_artifacts_and_logs_settings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "song.mp3").write_bytes(b"")
            config = root / "config.json"
            config.write_text(json.dumps({
                "input_audio": "song.mp3",
                "upscale": {"enabled": True, "target_width": 2560, "max_pass_scale": 2, "max_ai_passes": 3},
            }), encoding="utf-8")
            layout = SceneArtifactLayout(root)
            source = layout.scene_final_video(1)
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"video")
            plan = root / "plan.json"
            plan.write_text(json.dumps([{"scene": 1}]), encoding="utf-8")
            backend = FakeBackend()

            run_seedvr2(SeedVR2CompositionOptions(
                project_config_path=config,
                render_plan_path=plan,
                backend=backend,
                probe_size=lambda _path: (640, 360),
            ))

            manifest = json.loads((layout.scene_dir(1) / "upscale_manifest.json").read_text(encoding="utf-8"))
            final_exists = layout.scene_upscaled_video(1).is_file()

        self.assertEqual(2, len(backend.calls))
        self.assertEqual((1280, 720), backend.calls[0]["output_size"])
        self.assertEqual((2560, 1440), backend.calls[1]["output_size"])
        self.assertEqual("auto", manifest["strategy"])
        self.assertEqual(2, len(manifest["passes"]))
        self.assertTrue(final_exists)

    def test_run_seedvr2_reuses_existing_final_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "song.mp3").write_bytes(b"")
            config = root / "config.json"
            config.write_text(json.dumps({"input_audio": "song.mp3", "upscale": {"enabled": True, "target_width": 2560}}), encoding="utf-8")
            layout = SceneArtifactLayout(root)
            source = layout.scene_final_video(1)
            final = layout.scene_upscaled_video(1)
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"source")
            final.write_bytes(b"existing")
            plan = root / "plan.json"
            plan.write_text(json.dumps([{"scene": 1}]), encoding="utf-8")
            backend = FakeBackend()

            run_seedvr2(SeedVR2CompositionOptions(
                project_config_path=config,
                render_plan_path=plan,
                backend=backend,
                probe_size=lambda _path: (640, 360),
            ))

        self.assertEqual([], backend.calls)

    def test_run_seedvr2_rebuilds_existing_final_from_segments_without_backend_render(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "song.mp3").write_bytes(b"")
            config = root / "config.json"
            config.write_text(json.dumps({"input_audio": "song.mp3", "upscale": {"enabled": True, "target_width": 2560}}), encoding="utf-8")
            layout = SceneArtifactLayout(root)
            source = layout.scene_final_video(1)
            final = layout.scene_upscaled_video(1)
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"source")
            final.write_bytes(b"stale final")
            segment_one = layout.scene_dir(1) / "upscale_pass_01_segment_0001.mp4"
            segment_two = layout.scene_dir(1) / "upscale_pass_01_segment_0002.mp4"
            segment_one.write_bytes(b"segment one")
            segment_two.write_bytes(b"segment two")
            segment_list = layout.scene_dir(1) / "upscale_pass_01_segments.txt"
            segment_list.write_text(f"file '{segment_one}'\nfile '{segment_two}'\n", encoding="utf-8")
            plan = root / "plan.json"
            plan.write_text(json.dumps([{"scene": 1}]), encoding="utf-8")
            backend = FakeBackend()
            postprocessor = FakePostProcessor()

            with patch("feverslop.composition.seedvr2_pipeline.VideoPostProcessor", return_value=postprocessor):
                run_seedvr2(SeedVR2CompositionOptions(
                    project_config_path=config,
                    render_plan_path=plan,
                    backend=backend,
                    probe_size=lambda _path: (640, 360),
                    probe_duration=lambda _path: 11.08,
                    reporter=RecordingReporter(),
                ))

        self.assertEqual([], backend.calls)
        self.assertEqual((True, True, 24, 266), postprocessor.concat_calls[-1][2:])

    def test_run_seedvr2_processes_only_selected_scenes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "song.mp3").write_bytes(b"")
            config = root / "config.json"
            config.write_text(json.dumps({"input_audio": "song.mp3", "upscale": {"enabled": True}}), encoding="utf-8")
            layout = SceneArtifactLayout(root)
            for scene_number in (1, 3):
                source = layout.scene_final_video(scene_number)
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(b"video")
            plan = root / "plan.json"
            plan.write_text(json.dumps([{"scene": 1}, {"scene": 3}]), encoding="utf-8")
            backend = FakeBackend()

            run_seedvr2(SeedVR2CompositionOptions(
                project_config_path=config,
                render_plan_path=plan,
                backend=backend,
                probe_size=lambda _path: (640, 360),
                scene_numbers={3},
            ))

        self.assertEqual(1, len(backend.calls))
        self.assertEqual(3, backend.calls[0]["scene_number"])

    def test_run_seedvr2_reports_each_scene_and_pass(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "song.mp3").write_bytes(b"")
            config = root / "config.json"
            config.write_text(json.dumps({"input_audio": "song.mp3", "upscale": {"enabled": True, "target_width": 2560, "segment_duration_seconds": 100}}), encoding="utf-8")
            layout = SceneArtifactLayout(root)
            source = layout.scene_final_video(1)
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"video")
            plan = root / "plan.json"
            plan.write_text(json.dumps([{"scene": 1}]), encoding="utf-8")
            reporter = RecordingReporter()
            backend = FakeBackend()

            run_seedvr2(SeedVR2CompositionOptions(
                project_config_path=config,
                render_plan_path=plan,
                backend=backend,
                probe_size=lambda _path: (640, 360),
                probe_duration=lambda _path: 11.08,
                reporter=reporter,
            ))

        self.assertTrue(any("source" in message and "11.08s" in message for message in reporter.messages))
        self.assertTrue(any("vae_temporal_size=64" in message for message in reporter.messages))
        self.assertTrue(any("scene 1/1" in message and "starting" in message for message in reporter.messages))
        self.assertTrue(any("pass 1/2" in message and "complete" in message for message in reporter.messages))
        self.assertTrue(any("pass 2/2" in message and "complete" in message for message in reporter.messages))
        self.assertTrue(any("scene 1/1" in message and "complete" in message for message in reporter.messages))
        self.assertEqual(64, backend.calls[0]["settings"].vae_temporal_size)

    def test_run_seedvr2_segments_long_final_pass_before_concat(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "song.mp3").write_bytes(b"")
            config = root / "config.json"
            config.write_text(json.dumps({
                "input_audio": "song.mp3",
                "upscale": {
                    "enabled": True,
                    "target_width": 2560,
                    "max_pass_scale": 2,
                    "max_ai_passes": 2,
                    "segment_duration_seconds": 4,
                },
            }), encoding="utf-8")
            layout = SceneArtifactLayout(root)
            source = layout.scene_final_video(1)
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"video")
            plan = root / "plan.json"
            plan.write_text(json.dumps([{"scene": 1}]), encoding="utf-8")
            backend = FakeBackend()
            reporter = RecordingReporter()
            for index in (1, 2, 3):
                (layout.scene_dir(1) / f"upscale_pass_01_segment_{index:04d}.mp4").write_bytes(b"old segment")

            postprocessor = FakePostProcessor()
            with patch("feverslop.composition.seedvr2_pipeline.VideoPostProcessor", return_value=postprocessor):
                run_seedvr2(SeedVR2CompositionOptions(
                    project_config_path=config,
                    render_plan_path=plan,
                    backend=backend,
                    probe_size=lambda _path: (640, 360),
                    probe_duration=lambda _path: 11.08,
                    reporter=reporter,
                    skip_existing=False,
                ))
                final_exists = layout.scene_upscaled_video(1).is_file()

        self.assertEqual(4, len(backend.calls))
        segment_calls = backend.calls[1:]
        self.assertEqual([0.0, 4.0, 8.0], [call["settings"].trim_start_seconds for call in segment_calls])
        self.assertEqual([4.0, 4.0, 3.08], [call["settings"].trim_duration_seconds for call in segment_calls])
        self.assertTrue(final_exists)
        self.assertTrue(any("segment 3/3 complete" in message for message in reporter.messages))
        self.assertEqual((True, True, 24, 266), postprocessor.concat_calls[-1][2:])

    def test_run_seedvr2_finds_legacy_ltx_scene_clip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "song.mp3").write_bytes(b"")
            config = root / "config.json"
            config.write_text(json.dumps({"input_audio": "song.mp3", "upscale": {"enabled": True}}), encoding="utf-8")
            legacy_source = root / "output" / "movie" / "ltx_msr" / "scene_0001.mp4"
            legacy_source.parent.mkdir(parents=True, exist_ok=True)
            legacy_source.write_bytes(b"video")
            plan = root / "plan.json"
            plan.write_text(json.dumps([{"scene": 1}]), encoding="utf-8")
            backend = FakeBackend()

            run_seedvr2(SeedVR2CompositionOptions(
                project_config_path=config,
                render_plan_path=plan,
                backend=backend,
                probe_size=lambda _path: (640, 360),
            ))

        self.assertEqual(legacy_source, backend.calls[0]["source_video"])
