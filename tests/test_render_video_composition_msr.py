import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from feverslop.adapters.comfyui_msr_video_backend import ComfyUIMSRVideoRenderBackend
from feverslop.composition.render_video import RenderVideoCompositionOptions, build_render_video_scenes_use_case


class RenderVideoCompositionMSRTests(unittest.TestCase):
    def test_video_pipeline_ltx_msr_builds_msr_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            app_config = temp / "app_config.json"
            app_config.write_text(json.dumps({"comfyui": {"base_url": "http://127.0.0.1:8188"}}), encoding="utf-8")
            workflow = temp / "workflow.json"
            workflow.write_text("{}", encoding="utf-8")

            with patch("feverslop.composition.render_video.ComfyUIClient"):
                use_case = build_render_video_scenes_use_case(
                    RenderVideoCompositionOptions(
                        app_config_path=app_config,
                        workflow_path=workflow,
                        output_dir=temp / "out",
                        video_pipeline="ltx_msr",
                    )
                )

            self.assertIsInstance(use_case.backend, ComfyUIMSRVideoRenderBackend)
