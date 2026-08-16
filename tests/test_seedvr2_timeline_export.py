import json
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import unittest

from feverslop.composition.stage_runners import _run_timeline_export_stage
from feverslop.scene_artifacts import SceneArtifactLayout


class SeedVR2TimelineExportTests(unittest.TestCase):
    def test_mlt_export_writes_final_facefix_and_upscaled_projects(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "song.mp3").write_bytes(b"audio")
            config = root / "config.json"
            config.write_text(json.dumps({"input_audio": "song.mp3", "video": {"width": 640, "height": 360, "fps": 24}}), encoding="utf-8")
            layout = SceneArtifactLayout(root)
            plan = layout.base_plan
            plan.parent.mkdir(parents=True, exist_ok=True)
            plan.write_text(json.dumps([{"scene": 1, "duration_seconds": 1}]), encoding="utf-8")
            for path in (layout.scene_final_video(1), layout.scene_final_facefix_video(1), layout.scene_upscaled_video(1)):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"video")
            written = []
            state = Namespace(
                plan_for_next_step=plan,
                app_config_path=root / "app_config.json",
                context=Namespace(
                    project_config_path=config,
                    artifact_layout=layout,
                    ltx_dir=layout.scenes_dir,
                    project_output_dir=layout.output_dir,
                    project_file_stem="song",
                    input_audio=root / "song.mp3",
                ),
                args=Namespace(stages=[], timeline_format="mlt", skip_facefix=False),
            )

            def fake_export(**kwargs):
                written.append((Path(kwargs["output_path"]).name, Path(kwargs["clip_paths"][0]).name))
                return Path(kwargs["output_path"])

            with patch("feverslop.composition.stage_runners.export_render_plan_to_mlt", side_effect=fake_export):
                _run_timeline_export_stage(state)

        self.assertEqual(
            [("song.mlt", "final.mp4"), ("song_facefix.mlt", "final_facefix.mp4"), ("song_upscaled.mlt", "upscale_final.mp4")],
            written,
        )
