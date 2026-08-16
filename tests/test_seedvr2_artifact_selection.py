import json
import tempfile
import unittest
from pathlib import Path

from feverslop.composition.config_loader import collect_render_plan_scene_clips
from feverslop.scene_artifacts import SceneArtifactLayout


class SeedVR2ArtifactSelectionTests(unittest.TestCase):
    def test_upscaled_clip_is_preferred_when_present(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            layout = SceneArtifactLayout(root)
            plan = root / "plan.json"
            plan.write_text(json.dumps([{"scene": 1}]), encoding="utf-8")
            for path in (layout.scene_final_video(1), layout.scene_final_facefix_video(1), layout.scene_upscaled_video(1)):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"video")

            clips = collect_render_plan_scene_clips(plan, root / "legacy", layout=layout, prefer_upscaled=True)

        self.assertEqual(layout.scene_upscaled_video(1), clips[0])

    def test_upscale_selection_falls_back_to_facefix_then_final(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            layout = SceneArtifactLayout(root)
            plan = root / "plan.json"
            plan.write_text(json.dumps([{"scene": 1}]), encoding="utf-8")
            final = layout.scene_final_video(1)
            facefix = layout.scene_final_facefix_video(1)
            final.parent.mkdir(parents=True, exist_ok=True)
            final.write_bytes(b"video")

            self.assertEqual(final, collect_render_plan_scene_clips(plan, root / "legacy", layout=layout, prefer_upscaled=True)[0])
            facefix.write_bytes(b"video")
            self.assertEqual(facefix, collect_render_plan_scene_clips(plan, root / "legacy", layout=layout, prefer_upscaled=True)[0])
