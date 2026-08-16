import json
import tempfile
import unittest
from pathlib import Path

from feverslop.composition.seedvr2_pipeline import SeedVR2CompositionOptions, run_seedvr2
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


class SeedVR2PipelineTests(unittest.TestCase):
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
            config.write_text(json.dumps({"input_audio": "song.mp3", "upscale": {"enabled": True}}), encoding="utf-8")
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
